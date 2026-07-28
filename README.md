# Muonly

Deep-learning preselection of inner tracker tracks for CMS Phase-2 tracker muon
reconstruction. See [docs/overview.md](docs/overview.md) for the project goal
and [docs/getting-started.md](docs/getting-started.md) for the full guide.

## Recipes

### Install dependencies
Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python >= 3.12):
```bash
uv sync
```
Run all commands through `uv run`; no separate environment activation is needed.

### Sanity check
Verify the full pipeline on a small subset before launching a real run:
```bash
uv run python scripts/train.py mode=sanity-check
```

### Training
Configuration is composed by Hydra from `config/`. Swap config groups
(`model=`, `loss=`, `data=`, `paths=`, `mode=`) or override individual keys:
```bash
uv run python scripts/train.py exp=my-study run=baseline \
    model=latent_cross_attention model.model_dim=128 \
    optim.lr=1e-4 optim.max_epochs=100 data_load.batch_size=256
```
Outputs are written to `logs/<exp>/<run>/`.

### Monitor training logs with Aim UI
The Aim repository is the `logs/` directory:
```bash
uv run aim up --port <PORT>
```
On a remote server, forward the port over SSH.

### Predict and export
```bash
uv run python scripts/predict.py -c logs/<exp>/<run>/checkpoints/best.pt -s test
uv run python scripts/export.py -c logs/<exp>/<run>/checkpoints/best.pt
```
