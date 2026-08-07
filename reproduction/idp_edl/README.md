# IDP-EDL local reproduction entry point

This directory provides a local-only wrapper around the official source in
`external/IDP-EDL`. The official model classes and `LoRALinear` implementation
are imported without editing the external source. The wrapper fixes the
author's hard-coded paths, the FASTA batch tokenization call, and the broken
`np.save` call while keeping the official component forward methods and meta
fusion order.

The upstream implementation is available at
<https://github.com/joestarXjx/IDP-EDL> and is expected at
`external/IDP-EDL` for local inference.

## Environment

Use the prepared environment and the frozen direct dependencies:

```bash
cd /data8T/IDPs_DM3000Train
.conda/idp_edl/bin/python -m unittest discover -s reproduction/idp_edl/tests -t . -v
```

The audited environment is Python 3.8.19, torch 2.4.1+cu121, and
transformers 4.46.1. The ProtT5 files must be local; the inference entry does
not silently download from Hugging Face.

## Full T5 to encoder-only FP16 conversion

The conversion command is resumable and auditable. It accepts a single
`pytorch_model.bin`/`model.safetensors` or an indexed set of shards. It writes
only encoder keys required by `T5EncoderModel`, converts them to float16,
preserves `config.json` and tokenizer files, and writes
`conversion_manifest.json` with source and target SHA256 records. A partial
conversion is kept in `<output>.partial/conversion_progress.json` and can be
continued with the same command.

The default output remains under this reproduction directory, so it is safe to
prepare the command before the main download completes:

```bash
.conda/idp_edl/bin/python -m reproduction.idp_edl.convert \
  --source-dir external/ProtTrans/weights/prot_t5_xl_uniref50 \
  --output-dir reproduction/idp_edl/artifacts/prot_t5_xl_uniref50-enc-fp16
```

After the full checkpoint is complete, an explicit output path can be used by
the operator:

```bash
.conda/idp_edl/bin/python -m reproduction.idp_edl.convert \
  --source-dir external/ProtTrans/weights/prot_t5_xl_uniref50 \
  --output-dir external/ProtTrans/weights/prot_t5_xl_uniref50-enc-fp16
```

Use `--force` only when the source checkpoint changed or a partial output is
known to be invalid. The command is not run by the static test suite against
the 11.3 GB download. A small synthetic full T5 checkpoint is used to verify
key filtering, strict coverage, float16 output, HF loading, manifest hashes,
and repeat/resume behavior.

## Prediction and evaluation

The author sample pair is parsed by record order with an ID and length guard.
The supported benchmark selectors are `mxd494` (494 proteins), `sl329` (the
author-filtered 322 proteins), and `disorder723` (723 proteins). Label `2` is
written to the residue table but excluded from every metric. `max_length=1024`
matches the author code: the last token is EOS, so at most 1023 residues are
evaluated. Truncated residues are reported per protein and in the summary.

Run a no-model smoke check while ProtT5 is still downloading:

```bash
.conda/idp_edl/bin/python -m reproduction.idp_edl.cli \
  --smoke --dry-run --output-dir /tmp/idp_edl_smoke
```

Audit a dataset's parsing, labels, and truncation without loading ProtT5:

```bash
.conda/idp_edl/bin/python -m reproduction.idp_edl.cli \
  --dataset sl329 --dry-run --output-dir /tmp/idp_edl_sl329_audit
```

After the encoder directory is available, run actual inference. Runtime is
deliberately FP32 to match the official generator: the unmodified official
`IDPModel.forward` creates its initial GRU hidden state with default FP32
`torch.zeros`. Passing FP16 encoder activations would therefore fail with a
hidden/input dtype mismatch. The conversion output is FP16 on disk, but the
loader casts it to FP32 for the main evaluation.

```bash
.conda/idp_edl/bin/python -m reproduction.idp_edl.cli \
  --dataset mxd494 \
  --model-dir external/ProtTrans/weights/prot_t5_xl_uniref50-enc-fp16 \
  --dtype float32 \
  --batch-size 4 \
  --weights-dir external/IDP-EDL/model \
  --output-dir reproduction/idp_edl/outputs/mxd494
```

`--dtype auto` is retained as an explicit FP32 alias. `--dtype float16` is
rejected with an actionable error rather than silently changing the official
model. For a CPU smoke inference, use a local model and `--dtype float32`; the
full ProtT5 model is not expected to be practical on CPU. The same command accepts
`--dataset all`, `--dataset sl329`, `--dataset disorder723`, `--fasta`, or
`--sequence`.

The default evaluation batch size is 4, matching the author's public
prediction notebook. Keep this fixed for paper comparisons: the recurrent
generic branch can amplify floating-point differences caused by a different
padded batch shape.

Each run writes:

* `idp_edl_residue_predictions.tsv`: one row per retained residue, including
  original/model amino acid, label, prediction, score, and truncation count.
* `idp_edl_summary.tsv`: one row per selected dataset with residue counts and
  Sn, Sp, BACC, MCC, AUC, Fmax, and calibration threshold fields.
* `idp_edl_summary.json`: the same summaries plus the label-2 and EOS policy.

## Public prediction package

After the three `*_fp32_batch4` runs are present, validate every retained
residue against the author sample pair and rebuild the redacted public release:

```bash
.conda/idp_edl/bin/python scripts/package_idp_edl_reproduction.py
```

The command writes deterministic compressed prediction tables, recomputed
summaries, paper comparisons, and SHA256 provenance under
`results/reproduction/idp_edl/`. Public residue tables omit sequences,
amino-acid identities, headers, and reference labels.

## Reproduction assumptions

The external component checkpoints contain trainable parameters only. Frozen
ProtT5 base parameters come from the local T5 checkpoint. Before loading
`meta_predictor.pth`, all three component predictors are frozen exactly as in
the official `MetaPredictor`; only the six-to-two meta classifier remains
trainable. The conversion key check is derived from the local transformers
4.46.1 `T5EncoderModel` state-dict contract and is tested against a small
instantiated model. A full-load numerical comparison still requires the
completed local ProtT5 checkpoint.
