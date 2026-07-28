# Getting Started

This guide takes a new contributor from a fresh checkout to a first training
run, and then points to the reference documentation. It describes the tooling
only; the physics motivation is given in [Overview](overview.md).

## What the project does

The project trains a per-track binary classifier that preselects inner tracker
tracks before the expensive track-to-muon-system extrapolation in
`MuonIdProducer`. A track is positive if the corresponding post-arbitration
muon has its `TrackerMuon` bit set. The primary figure of merit is the true
negative rate at a true positive rate of at least 99.9%
(`tnr_at_tpr_0p9999`); see [Primary Evaluation Metric](metric.md).

## Environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/); Python 3.12 or
newer is required.

```bash
uv sync
```

Run every command through `uv run`, which resolves the project environment
without a separate activation step. Training assumes a CUDA device
(`torch.device: cuda:0` in `config/torch/default.yaml`).

## Data

Training uses three HDF5 splits (`train.h5`, `val.h5`, `test.h5`) produced from
DeepMuonReco ntuples. The directory is selected by a `paths` preset:

| Preset | `data_dir` |
| --- | --- |
| `khu` (default) | `/users/hep/joshin/store/muonly/dataset/` |
| `uos` | `/home/joshin/workspace-gate/DeepMuonReco/DeepMuonReco/data/mu2030pu/` |

Select another preset with `paths=uos`, or point to an arbitrary directory with
`paths.data_dir=/my/dir`. See [Dataset](data/data.md) for sample contents and
[Data Format](data/data-format.md) for tensor shapes and preprocessing.

## First run: sanity check

```bash
uv run python scripts/train.py mode=sanity-check
```

This trains for 2 epochs on 1024 training and 1024 validation events with batch
size 256, exercising the full pipeline in a few minutes. It must finish without
error before any full run is started.

## Full training run

```bash
uv run python scripts/train.py exp=my-study run=baseline optim.max_epochs=100
```

Configuration is composed by [Hydra](https://hydra.cc/) from `config/`.
`scripts/train.py` loads `config/no-hit.yaml`, which extends
`config/default.yaml` and disables the RPC and GEM hit inputs. Defaults are
`model=latent_cross_attention`, `data=mu2030pu`, `loss=focal`, `paths=khu`.

Two override styles are available:

- **Config groups**, which swap a whole file:
  `model=vanilla_transformer`, `loss=bce`, `paths=uos`, `mode=dev`.
- **Dotted keys**, which override single values:
  `model.model_dim=128`, `optim.lr=1e-4`, `data_load.batch_size=256`,
  `torch.seed=20260710`.

Set `exp` and `run` explicitly to obtain a readable run directory; otherwise
`run` defaults to a timestamp plus a random slug.

## Run outputs

Each run writes to `logs/<exp>/<run>/`. This directory is not tracked by git.

| Path | Content |
| --- | --- |
| `config.yaml` | Fully resolved configuration; the reproducibility record of the run. |
| `checkpoints/best.pt` | Best checkpoint, selected on validation loss. |
| `results/best/val.json` | Final validation metrics of the best checkpoint (AUROC and others). |
| `results/best/sas.json` | TNR and score threshold at TPR = 0.99, 0.999, 0.9999, 0.99999. |
| `results/best/*.png`, `*.pdf` | ROC curve, and efficiency and rejection versus track `pT`. |
| `model-summary.txt` | Layer summary and parameter counts. |
| `hydra.log` | Job log. |
| `memory.csv`, `cuda-memory.csv` | Host and device memory traces. |

## Monitoring

Metrics are tracked with [Aim](https://aimstack.io/); the Aim repository is the
`logs/` directory and the Aim experiment name is `exp`. From the repository
root:

```bash
uv run aim up --port <PORT>
```

On a remote machine, forward the port over SSH. The validation quantity to
watch is `tnr_at_tpr_0p9999`.

## After training

```bash
# per-track scores written to HDF5
uv run python scripts/predict.py -c logs/<exp>/<run>/checkpoints/best.pt -s test

# ONNX export with preprocessing baked into the graph
uv run python scripts/export.py -c logs/<exp>/<run>/checkpoints/best.pt
```

Export details are documented in [ONNX Export](onnx.md).

## Known issue

`torch.sdpa_backend` defaults to `math` rather than the optimized kernel. The
optimized bfloat16 SDPA CUDA path corrupts validation targets on the RTX 5090;
see [Invalid target value 16777216](issues/target-16777216.md).

## Where to read next

| Document | Purpose |
| --- | --- |
| [overview.md](overview.md) | Project goal, proposed reconstruction flow, current status. |
| [metric.md](metric.md) | Definition and computation of TNR at TPR >= 99.9%. |
| [data/data-format.md](data/data-format.md) | Model input and output tensors, feature order, masks, preprocessing. |
| [data/data.md](data/data.md) | Training sample and object multiplicity statistics. |
| [loss.md](loss.md) | Config-driven loss framework and auxiliary terms. |
| [study/loss.md](study/loss.md) | Loss ablation study and its conclusions. |
| [dev/plan.md](dev/plan.md) | Prioritized plan for improving model performance. |
| [data/muon-id-producer.md](data/muon-id-producer.md) | Existing tracker muon reconstruction and label definition. |
| [data/ntuplizer.md](data/ntuplizer.md) | Ntuple production and ROOT-to-HDF5 conversion. |
| [onnx.md](onnx.md) | ONNX export for downstream CMSSW inference. |
