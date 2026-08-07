"""Safe imports of the official IDP-EDL model classes.

The official files stay under ``external/IDP-EDL`` and are never edited.  The
loader imports their classes under private module names and uses the official
LoRALinear implementation and model forward methods.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class ModelLoadError(RuntimeError):
    pass


_API_CACHE: Dict[str, SimpleNamespace] = {}


def validate_runtime_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Enforce the official generator's FP32 runtime contract.

    The official IDPModel creates its initial hidden state with a default
    FP32 ``torch.zeros``. Running the ProtT5 stack in FP16 therefore causes a
    dtype mismatch in the unmodified forward method.
    """

    if dtype == torch.float16:
        raise ModelLoadError(
            "FP16 runtime is not supported by the unmodified official IDPModel.forward "
            "(its h0 is created in FP32); use --dtype float32. FP16 is supported "
            "for encoder checkpoint storage/conversion only."
        )
    return torch.float32


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ModelLoadError("cannot import official module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_api(external_root: Path) -> SimpleNamespace:
    root = Path(external_root).resolve()
    key = str(root)
    if key in _API_CACHE:
        return _API_CACHE[key]
    if not (root / "models.py").is_file():
        raise ModelLoadError("official IDP-EDL source is missing: {}".format(root))

    config = _load_module("idp_edl_official_config", root / "config.py")
    layers = _load_module("idp_edl_official_layers", root / "layers.py")
    # models.py uses the official file's absolute ``from layers import ...``.
    previous_layers = sys.modules.get("layers")
    sys.modules["layers"] = layers
    try:
        models = _load_module("idp_edl_official_models", root / "models.py")
    finally:
        if previous_layers is None:
            sys.modules.pop("layers", None)
        else:
            sys.modules["layers"] = previous_layers

    api = SimpleNamespace(
        ClassConfig=config.ClassConfig,
        LoRAConfig=config.LoRAConfig,
        IDPModel=models.IDPModel,
        SDRModel=models.SDRModel,
        LDRModel=models.LDRModel,
        ClassiferModel=models.ClassiferModel,
        LoRALinear=layers.LoRALinear,
    )
    _API_CACHE[key] = api
    return api


def apply_official_lora(model: nn.Module, lora_config) -> nn.Module:
    """The same module traversal as official ``generator.py``."""

    for module_name, module in dict(model.named_modules()).items():
        if re.fullmatch(lora_config.lora_modules, module_name):
            for child_name, layer in dict(module.named_children()).items():
                if re.fullmatch(lora_config.lora_layers, child_name):
                    if not isinstance(layer, nn.Linear):
                        raise ModelLoadError(
                            "official LoRA target is not Linear: {} ({})".format(child_name, type(layer))
                        )
                    setattr(
                        module,
                        child_name,
                        _API_CURRENT.LoRALinear(
                            layer,
                            lora_config.lora_rank,
                            lora_config.lora_scaling_rank,
                            lora_config.lora_init_scale,
                        ),
                    )
    return model


_API_CURRENT = SimpleNamespace(LoRALinear=None)


def _set_requires_grad_like_official(model: nn.Module, lora_config) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if re.fullmatch(lora_config.trainable_param_names, name):
            parameter.requires_grad = True


def freeze_components_for_meta(meta: nn.Module) -> None:
    """Match official MetaPredictor: only its six-to-two head is trainable."""

    for component_name in ("idp_model", "sdr_model", "ldr_model"):
        component = getattr(meta, component_name)
        for parameter in component.parameters():
            parameter.requires_grad = False


def _load_parameter_file(model: nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    if not checkpoint_path.is_file():
        raise ModelLoadError("component checkpoint is missing: {}".format(checkpoint_path))
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ModelLoadError("checkpoint is not a state dict: {}".format(checkpoint_path))
    model_keys = set(model.state_dict().keys())
    unknown = sorted(set(state).difference(model_keys))
    if unknown:
        raise ModelLoadError(
            "checkpoint has {} unexpected keys; first keys: {}".format(len(unknown), unknown[:5])
        )
    incompatible = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(trainable.difference(state))
    if missing_trainable:
        raise ModelLoadError(
            "checkpoint misses {} trainable parameters; first keys: {}".format(
                len(missing_trainable), missing_trainable[:5]
            )
        )
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": int(len(state)),
        "missing_frozen_or_buffers": int(len(incompatible.missing_keys)),
        "unexpected_keys": int(len(incompatible.unexpected_keys)),
    }


def _materialize_meta_module(module: nn.Module) -> nn.Module:
    if any(parameter.is_meta for parameter in module.parameters()):
        module = module.to_empty(device="cpu")
        for child in module.modules():
            if isinstance(child, nn.BatchNorm1d):
                if child.running_mean is not None:
                    child.running_mean.zero_()
                if child.running_var is not None:
                    child.running_var.fill_(1.0)
                if child.num_batches_tracked is not None:
                    child.num_batches_tracked.zero_()
    return module


def _build_component(
    api: SimpleNamespace,
    component_class,
    model_dir: Path,
    checkpoint_path: Path,
    dtype: Optional[torch.dtype],
) -> tuple:
    kwargs = {"local_files_only": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    try:
        encoder = T5EncoderModel.from_pretrained(str(model_dir), **kwargs)
    except Exception as exc:
        raise ModelLoadError(
            "cannot load local ProtT5 encoder from {}. The full checkpoint or an encoder conversion is required: {}".format(
                model_dir, exc
            )
        )
    config = api.ClassConfig(encoder.config)
    # Construct the official head on meta to avoid allocating a second 1.2B
    # parameter T5 encoder before replacing it with the loaded local encoder.
    try:
        with torch.device("meta"):
            component = component_class(config)
        component = _materialize_meta_module(component)
    except Exception:
        component = component_class(config)
    component.encoder = encoder
    global _API_CURRENT
    _API_CURRENT = api
    component = apply_official_lora(component, api.LoRAConfig())
    _set_requires_grad_like_official(component, api.LoRAConfig())
    report = _load_parameter_file(component, checkpoint_path)
    return component, report


def load_idp_edl(
    model_dir: Path,
    weights_dir: Path,
    external_root: Path,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> SimpleNamespace:
    """Load the local tokenizer, three official components, and meta head."""

    dtype = validate_runtime_dtype(dtype)
    model_dir = Path(model_dir).resolve()
    weights_dir = Path(weights_dir).resolve()
    api = load_official_api(Path(external_root))
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            str(model_dir), do_lower_case=False, local_files_only=True
        )
    except Exception as exc:
        raise ModelLoadError("cannot load local ProtT5 tokenizer from {}: {}".format(model_dir, exc))

    generic, generic_report = _build_component(
        api, api.IDPModel, model_dir, weights_dir / "generic_predictor.pth", dtype
    )
    sdr, sdr_report = _build_component(
        api, api.SDRModel, model_dir, weights_dir / "sdr_predictor.pth", dtype
    )
    ldr, ldr_report = _build_component(
        api, api.LDRModel, model_dir, weights_dir / "ldr_predictor.pth", dtype
    )

    # This is the post-replacement module graph produced by official
    # MetaPredictor, without allocating its three discarded temporary T5 stacks.
    meta = object.__new__(api.ClassiferModel)
    nn.Module.__init__(meta)
    meta.num_labels = 2
    meta.ldr_model = ldr
    meta.sdr_model = sdr
    meta.idp_model = generic
    meta.classifier = nn.Linear(6, 2)
    freeze_components_for_meta(meta)
    meta_report = _load_parameter_file(meta, weights_dir / "meta_predictor.pth")

    base_dtype = next(meta.idp_model.encoder.parameters()).dtype
    meta = meta.to(device=device, dtype=base_dtype)
    meta.eval()
    for parameter in meta.parameters():
        parameter.requires_grad = False
    return SimpleNamespace(
        model=meta,
        tokenizer=tokenizer,
        device=device,
        dtype=base_dtype,
        reports={
            "generic": generic_report,
            "sdr": sdr_report,
            "ldr": ldr_report,
            "meta": meta_report,
        },
    )
