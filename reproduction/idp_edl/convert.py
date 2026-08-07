"""Recoverable conversion of a full ProtT5 checkpoint to encoder-only FP16.

The command is intentionally local-only.  It reads a Hugging Face T5
checkpoint, filters the exact key set required by ``T5EncoderModel``, writes
safe-tensors output, preserves tokenizer/config files, and records SHA256
hashes for both source and target files.  A ``.partial`` directory and a
progress JSON make interrupted shard conversions restartable.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch


class ConversionError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, object]:
    path = Path(path).resolve()
    return {"path": str(path), "size": int(path.stat().st_size), "sha256": sha256_file(path)}


def expected_encoder_keys(config: Mapping[str, object]) -> Set[str]:
    """Derive the exact T5EncoderModel state-dict key set from config.

    The key construction is checked in tests against an instantiated small
    ``T5EncoderModel``.  It avoids constructing a 1.2B-parameter random model
    merely to validate a full checkpoint's key coverage.
    """

    layers = int(config["num_layers"])
    keys = {"shared.weight", "encoder.embed_tokens.weight", "encoder.final_layer_norm.weight"}
    gated = bool(config.get("is_gated_act", False)) or str(config.get("feed_forward_proj", "")).startswith("gated-")
    d_ff_names = ("wi_0.weight", "wi_1.weight") if gated else ("wi.weight",)
    for index in range(layers):
        prefix = "encoder.block.{}.layer.".format(index)
        keys.update(
            {
                prefix + "0.SelfAttention.q.weight",
                prefix + "0.SelfAttention.k.weight",
                prefix + "0.SelfAttention.v.weight",
                prefix + "0.SelfAttention.o.weight",
                prefix + "0.layer_norm.weight",
                prefix + "1.layer_norm.weight",
                prefix + "1.DenseReluDense.wo.weight",
            }
        )
        keys.update(prefix + "1.DenseReluDense." + name for name in d_ff_names)
        if index == 0:
            keys.add(prefix + "0.SelfAttention.relative_attention_bias.weight")
    return keys


def _weight_files_from_index(source_dir: Path, index_path: Path) -> List[Path]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ConversionError("invalid checkpoint index {}: {}".format(index_path, exc))
    if not isinstance(weight_map, dict):
        raise ConversionError("checkpoint index has no mapping: {}".format(index_path))
    names = sorted(set(str(name) for name in weight_map.values()))
    files = [source_dir / name for name in names]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ConversionError("checkpoint index references missing shards: {}".format(missing[:5]))
    return files


def discover_weight_files(source_dir: Path) -> List[Path]:
    source_dir = Path(source_dir)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = source_dir / index_name
        if index_path.is_file():
            return _weight_files_from_index(source_dir, index_path)
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = source_dir / name
        if path.is_file():
            return [path]
    candidates = sorted(
        list(source_dir.glob("model-*.safetensors"))
        + list(source_dir.glob("pytorch_model-*.bin"))
    )
    if candidates:
        return candidates
    raise ConversionError(
        "no full checkpoint found in {}; expected model.safetensors, pytorch_model.bin, or an index".format(
            source_dir
        )
    )


def _load_bin(path: Path) -> Mapping[str, torch.Tensor]:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ConversionError("checkpoint shard is not a state dict: {}".format(path))
    return state


def _select_encoder_tensors(path: Path, required: Set[str]) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors" or path.name.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ConversionError("safetensors is required for {}: {}".format(path, exc))
        selected = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in required:
                    selected[key] = handle.get_tensor(key)
        return selected
    state = _load_bin(path)
    return {key: value for key, value in state.items() if key in required}


def _as_float16_tensors(tensors: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise ConversionError("encoder key {} is not a tensor".format(key))
        if not torch.is_floating_point(value):
            raise ConversionError("encoder key {} is not floating point".format(key))
        # clone breaks the shared storage between shared.weight and
        # encoder.embed_tokens.weight so safe-tensors can save both aliases.
        result[key] = value.detach().to(dtype=torch.float16).contiguous().clone()
    return result


def _write_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ConversionError("safetensors is required to write encoder output: {}".format(exc))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(path), metadata={"format": "pt", "dtype": "float16"})


def _read_safetensor_keys(path: Path) -> Tuple[Set[str], int]:
    from safetensors import safe_open

    keys = set()
    total_size = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            keys.add(key)
            if tensor.dtype != torch.float16:
                raise ConversionError("target tensor {} is not float16 in {}".format(key, path))
            total_size += int(tensor.numel() * tensor.element_size())
    return keys, total_size


def _source_metadata_files(source_dir: Path, weight_files: Sequence[Path]) -> List[Path]:
    names = {path.name for path in weight_files}
    names.update(
        {
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "added_tokens.json",
            "generation_config.json",
        }
    )
    return sorted(path for path in (source_dir / name for name in names) if path.is_file())


def _source_signature(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConversionError("invalid JSON file {}: {}".format(path, exc))


def _copy_metadata(source_dir: Path, staging: Path) -> Dict[str, object]:
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise ConversionError("source config.json is missing: {}".format(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["T5EncoderModel"]
    config["torch_dtype"] = "float16"
    config.pop("_name_or_path", None)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied = ["config.json"]
    for name in (
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "added_tokens.json",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(str(source), str(staging / name))
            copied.append(name)
    return config


def _target_file_records(directory: Path) -> List[Dict[str, object]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"conversion_manifest.json", "conversion_progress.json"}:
            records.append(
                {
                    "path": str(path.relative_to(directory)),
                    "size": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return records


def _verify_target(staging: Path, required: Set[str]) -> Tuple[Set[str], int, Dict[str, str]]:
    output_files = sorted(staging.glob("*.safetensors"))
    if not output_files:
        raise ConversionError("conversion produced no encoder safetensors")
    seen = set()
    total_size = 0
    weight_map = {}
    for path in output_files:
        keys, size = _read_safetensor_keys(path)
        duplicate = seen.intersection(keys)
        if duplicate:
            raise ConversionError("duplicate encoder keys in target: {}".format(sorted(duplicate)[:5]))
        seen.update(keys)
        total_size += size
        for key in keys:
            weight_map[key] = path.name
    missing = sorted(required.difference(seen))
    extra = sorted(seen.difference(required))
    if missing or extra:
        raise ConversionError(
            "strict encoder key check failed; missing={} extra={}".format(missing[:5], extra[:5])
        )
    return seen, total_size, weight_map


def convert_checkpoint(
    source_dir: Path,
    output_dir: Path,
    force: bool = False,
    resume: bool = True,
) -> Dict[str, object]:
    """Convert a local full checkpoint and return its completed manifest."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise ConversionError("source directory does not exist: {}".format(source_dir))
    weight_files = discover_weight_files(source_dir)
    metadata_files = _source_metadata_files(source_dir, weight_files)
    source_records = [_file_record(path) for path in metadata_files]
    signature = _source_signature(source_records)
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    required = expected_encoder_keys(config)

    final_manifest_path = output_dir / "conversion_manifest.json"
    final_manifest = _load_json(final_manifest_path)
    if final_manifest and final_manifest.get("status") == "complete":
        if final_manifest.get("source_signature") == signature:
            expected_targets = final_manifest.get("target_files", [])
            valid = True
            for record in expected_targets:
                path = output_dir / str(record["path"])
                if not path.is_file() or sha256_file(path) != record.get("sha256"):
                    valid = False
                    break
            if valid:
                return final_manifest
        if not force:
            raise ConversionError(
                "output already exists with a different or invalid source; use --force: {}".format(output_dir)
            )

    staging = output_dir.with_name(output_dir.name + ".partial")
    progress_path = staging / "conversion_progress.json"
    if force:
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        if staging.exists():
            shutil.rmtree(str(staging))
    elif not resume and staging.exists():
        # --no-resume means a clean rebuild; otherwise stale shard files could
        # be mistaken for current encoder output during the strict key check.
        shutil.rmtree(str(staging))
    progress = _load_json(progress_path) if resume else None
    if progress and progress.get("source_signature") != signature:
        raise ConversionError("partial conversion belongs to a different source; use --force: {}".format(staging))
    if progress is None:
        progress = {
            "status": "partial",
            "source_signature": signature,
            "source_records": source_records,
            "required_key_count": len(required),
            "completed_shards": {},
        }
        staging.mkdir(parents=True, exist_ok=True)
        _atomic_json(progress_path, progress)

    completed = progress.setdefault("completed_shards", {})
    for index, weight_path in enumerate(weight_files, 1):
        shard_key = str(weight_path.resolve())
        entry = completed.get(shard_key)
        if entry and entry.get("output"):
            output_path = staging / str(entry["output"])
            if output_path.is_file() and entry.get("sha256") == sha256_file(output_path):
                continue
        selected = _select_encoder_tensors(weight_path, required)
        output_name = (
            "model.safetensors" if len(weight_files) == 1 else "encoder-shard-{:05d}.safetensors".format(index)
        )
        if selected:
            _write_safetensors(staging / output_name, _as_float16_tensors(selected))
            output_entry = {
                "output": output_name,
                "keys": sorted(selected),
                "sha256": sha256_file(staging / output_name),
            }
        else:
            output_entry = {"output": None, "keys": [], "sha256": None}
        completed[shard_key] = output_entry
        progress["completed_shards"] = completed
        _atomic_json(progress_path, progress)

    seen, total_size, weight_map = _verify_target(staging, required)
    _copy_metadata(source_dir, staging)
    output_safetensors = sorted(staging.glob("*.safetensors"))
    if len(output_safetensors) > 1:
        _atomic_json(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
    # The progress file must not become part of the final HF directory.
    if progress_path.exists():
        progress_path.unlink()
    target_records = _target_file_records(staging)
    manifest = {
        "format": "idp-edl-prott5-encoder-conversion-v1",
        "status": "complete",
        "source_dir": str(source_dir),
        "source_signature": signature,
        "source_files": source_records,
        "target_dir": str(output_dir),
        "target_files": target_records,
        "required_encoder_key_count": len(required),
        "written_encoder_key_count": len(seen),
        "encoder_key_check": "pass",
        "dtype": "float16",
        "total_encoder_bytes": total_size,
        "resume_supported": True,
    }
    _atomic_json(staging / "conversion_manifest.json", manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not force:
            raise ConversionError("output directory appeared during conversion: {}".format(output_dir))
        shutil.rmtree(str(output_dir))
    os.replace(str(staging), str(output_dir))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("external/ProtTrans/weights/prot_t5_xl_uniref50"),
        help="directory containing the full ProtT5 checkpoint and tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction/idp_edl/artifacts/prot_t5_xl_uniref50-enc-fp16"),
        help="encoder-only HF output directory; default stays inside reproduction/idp_edl",
    )
    parser.add_argument("--force", action="store_true", help="discard an invalid final/partial conversion")
    parser.add_argument("--no-resume", action="store_true", help="ignore a partial conversion directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = convert_checkpoint(
            args.source_dir,
            args.output_dir,
            force=args.force,
            resume=not args.no_resume,
        )
    except ConversionError as exc:
        print("ERROR: {}".format(exc))
        return 2
    print(
        "conversion complete: {} encoder keys, {} target files; manifest={}".format(
            manifest["written_encoder_key_count"], len(manifest["target_files"]), args.output_dir / "conversion_manifest.json"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
