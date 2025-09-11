# FASO: Feedback Arc Set Optimization in LLM Alignment

## Running Modules Consistently

- Use module execution with absolute imports to avoid `sys.path` hacks and directory-dependent behavior.
- Examples:
  - Train DPO: `python -m src.train.dpo config/experiment_cyclic.yaml <dataset_json>`
  - Generate outputs: `python -m src.prepare_data.generate_output config/experiment_cyclic.yaml`
  - Preprocess comparisons: `python -m src.prepare_data.preprocess_comparisons config/experiment_cyclic.yaml <gen_filename> [-o <cached.jsonl>]`
  - Build labeled data: `python -m src.prepare_data.prepare_data config/experiment_cyclic.yaml <gen_filename> <comp_filename>`
  - Calculate win rate: `python -m src.validate.calculate_win_rate config/experiment_cyclic.yaml <pairs_path> <out_path>`

Notes:
- Ensure the repo root is your CWD when running `python -m ...` so the `src` package is discoverable.
- Environment variables from `.env` (e.g., `PROJECT_ROOT`, paths) are still respected.
