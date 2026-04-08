# Repository Guidelines
**Always reply in Chinese!**

## Project Structure & Module Organization
- `app.py` provides the Gradio-based GPU monitor UI for real-time visualization.
- `monitor/` contains the reusable monitoring core, event definitions, and framework integrations.
- `run.py` is the generic smoke/demo runner for validating event-to-GPU alignment.
- `plugin.py` and `plugin_rollout.py` are compatibility shims that forward to optional integrations.
- `data/` and `logs/` store runtime artifacts such as CSV outputs.

## Build, Test, and Development Commands
- Run the monitoring UI locally:
  - `python app.py`
  - Requires a Python 3.12 environment with `gradio`, `pandas`, `plotly`, `python-dotenv`, plus `amdsmi` if AMD or `pynvml` if NVIDIA.
- Run the generic smoke test:
  - `python run.py --steps 3`
- Minimal module smoke test:
  - `python -m monitor`

## Coding Style & Naming Conventions
- Keep it simple, stupid. Only add comment if needed
- Use 4-space indentation and PEP 8 naming.
- Prefer `snake_case` for functions/variables and `UPPER_SNAKE_CASE` for constants.
- UI labels can be bilingual; keep them short and consistent with nearby sections.

## Commit & Pull Request Guidelines
- Git history is not present in this checkout; follow your team’s commit conventions.
- Keep framework-specific adapters optional; do not reintroduce hard imports into the monitoring core.

## Architecture & Runtime Notes
- The system aims to correlate GRPO phases with GPU telemetry at ms granularity.
- Only the rank-0 process should start the UI in multi-GPU training runs.
- Monitoring data is buffered in memory and periodically flushed to CSV for replay.
- Prefer integrating new training frameworks through `monitor/integrations/` instead of changing the core monitor.
