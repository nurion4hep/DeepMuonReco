## Recipes
1. Install requirements via micromamba
```bash
micromamba create -y -f ./environment.yaml
```

2. Activate the environment
source a shell script depending on your shell
```bash
source setup.sh
# or
source setup.fish
```

3. Do sanity-check
```bash
./train.py
```

4. Open Aim UI
```bash
aim up --port <PORT>
```
If you are working on a remote server, you need to set up port forwarding

## Development

### Code Quality

This project uses [Ruff](https://github.com/astral-sh/ruff) for code linting and formatting. 

To run linting locally:
```bash
pip install ruff
ruff check .
```

To run formatting checks:
```bash
ruff format --check .
```

To automatically fix formatting issues:
```bash
ruff format .
```

The linting and formatting checks are automatically run in GitHub Actions on all pull requests.
