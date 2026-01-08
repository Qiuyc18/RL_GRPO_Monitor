# Repository Guidelines
**Always reply in Chinese!**

## Project Structure & Module Organization
- `app.py` provides the Gradio-based GPU monitor UI for real-time visualization.
- `monitor_gpu.py` collects NVML metrics and maintains in-memory buffers plus CSV logging.
- `data/` stores runtime artifacts such as `data/gpu_metrics.csv`.
- `ms-swift/` is a vendored ms-swift framework used for GRPO/RLHF training and callbacks.

## Build, Test, and Development Commands
- Run the monitoring UI locally:
  - `python app.py`
  - Requires a Python 3.12 environment with `gradio`, `pandas`, `amdsmi` if AMD or `pynvml` if NVIDIA.
- Run ms-swift’s bundled test runner (GPU-heavy):
  - `python ms-swift/tests/run.py`
- Targeted tests (if dependencies are installed):
  - `pytest ms-swift/tests/train/test_grpo.py`

## Coding Style & Naming Conventions
- Keep it simple, stupid. Only add comment if needed
- Use 4-space indentation and PEP 8 naming.
- Prefer `snake_case` for functions/variables and `UPPER_SNAKE_CASE` for constants.
- UI labels can be bilingual; keep them short and consistent with nearby sections.

## Commit & Pull Request Guidelines
- Git history is not present in this checkout; follow your team’s commit conventions.
- Upstream contributions to ms-swift should follow `ms-swift/CONTRIBUTING.md`.

## Architecture & Runtime Notes
- The system aims to correlate GRPO phases with GPU telemetry at ms granularity.
- Only the rank-0 process should start the UI in multi-GPU training runs.
- Monitoring data is buffered in memory and periodically flushed to CSV for replay.
