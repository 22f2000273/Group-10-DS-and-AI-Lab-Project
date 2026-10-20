# team_10_dsai_iitm

## Project overview
This repository follows a reusable Data Science and AI project structure to support development from ideation to deployment.

## Problem statement
Create and maintain a clean, scalable repository layout for project code, data, reports, experiments, deployment assets, and tests.

## Features
- Clear separation of source code, data, docs, experiments, and outputs
- Placeholder structure for milestone reports and final artifacts
- Dedicated application layer (`app/`) separate from core logic (`src/`)
- Config-first workflow using files in `configs/`

## Architecture diagram
- See `docs/Architecture_Diagram.png`

## Dataset details
- Dataset guidance is documented in `data/README.md`
- Large datasets should not be committed; keep only small samples

## Model(s) used
- Model-related implementation belongs in `src/models/`
- Trained checkpoints belong in `models/checkpoints/`

## Installation steps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running instructions
- Use notebooks in `notebooks/` for exploration
- Move stable code into `src/`
- Put deployment interfaces under `app/`

## Results
- Store generated predictions, reports, figures, and logs under `outputs/`

## Demo screenshots
- Add demo screenshots in `outputs/figures/` and reference them here

## Team members
- Add team member names and roles here

## References
- Add external references, papers, and datasets here
