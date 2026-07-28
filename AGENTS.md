# Repository Guidelines

## Documentations
Read the following documentation files for detailed information about the project if needed:

- `docs/overview.md`: provide an overview of the project
- `docs/data/muon-id-producer.md` provides details about `MuonIdProducer`, which performs tracker muon reconstruction and provides labels for training and evaluation.
- `docs/data/ntuplizer.md` describes DeepMuonReco ntuple production, branch contents, ROOT-to-HDF5 conversion, and dataset integration.
- `docs/data/data.md` lists the Phase-2 SingleMu training sample and per-event track/segment multiplicity statistics.
- `docs/data/data-format.md` describes model input/output tensor shapes, dtypes, feature order, masks, and preprocessing (reference for ONNX export).
- `docs/metric.md` defines the primary evaluation metric (TNR at TPR >= 99.9%), its `BinarySpecificityAtSensitivity` computation, and the evaluation procedure.
- `docs/loss.md` describes the config-driven loss framework (`config/loss/*.yaml`, `muonly.nn.losses`): focal / asymmetric-focal criteria and batch-level auxiliary terms for hard-positive emphasis.
- `docs/study/loss.md` defines the loss-function ablation study: phased run matrix (criterion → aux terms → pos_weight → seeds), commands, and result tables.
- `docs/onnx.md` describes ONNX export (`scripts/export.py`, `Phase2NoHitModelWrapper`) with preprocessing baked into the graph.
- `docs/dev/plan.md` lays out the prioritized plan to improve `LatentCrossAttentionModel` TNR@TPR≥99.9%: loss alignment, richer inputs/embeddings, transformer depth/width tuning, training recipe, and experiment order.
- `docs/issues/target-16777216.md` documents the invalid `16777216` (2^24) binary-target failure in validation, traced to the optimized `bfloat16` SDPA CUDA path on the RTX 5090, with float32 / math-SDPA workarounds.
