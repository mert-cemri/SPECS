# SPECS: SPECulative test time Scaling

## Setup

```bash
bash install.sh
```

## Serving Models

Start the draft model, target model, and PRM in separate terminals:

```bash
# Usage: bash scripts/serve_<model>.sh [CUDA_DEVICE] [MODEL] [PORT]
bash scripts/serve_draft_model.sh 0
bash scripts/serve_target_model.sh 1
bash scripts/serve_prm.sh 2
```

## Run

```bash
bash run.sh
```

Or run directly with a recipe:

```bash
python scripts/test_time_compute.py recipes/specalign_rejection_amc.yaml \
    --draft_model_ip_address="http://localhost:12549/v1" \
    --target_model_ip_address="http://localhost:12545/v1" \
    --prm_ip_address="http://localhost:12550/v1"
```

## Project Structure

```
SPECS/
├── src/sal/              # Core library
│   ├── search/           # Search algorithms
│   │   ├── specalign_rejection.py  # SpecAlign rejection sampling
│   │   ├── beam_search.py          # Speculative beam search baseline
│   │   └── utils.py                # Shared search utilities
│   ├── utils/            # Utility functions
│   │   ├── data.py       # Dataset loading and saving
│   │   ├── score.py      # Scoring and evaluation
│   │   ├── math.py       # Math answer comparison
│   │   └── eval_utils.py # Evaluation utilities
│   ├── models/           # Model wrappers
│   └── config.py         # Configuration dataclass
├── scripts/              # Entry points and serving scripts
├── recipes/              # YAML configurations for experiments
├── external/             # Third-party evaluation packages
└── pyproject.toml
```
