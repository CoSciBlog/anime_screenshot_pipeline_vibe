# Changelog

## Unreleased - 2026-05-24

- Added the local Gradio Frame Lab interface on fixed port `7866`.
- Added independently selectable pipeline stages, per-stage settings with help text, TOML profile save/load, and a named profile execution API.
- Redesigned the UI with separate light/dark styling, per-stage setting tabs, and English interface text.
- Added `Stop pipeline` and `Shut down server` controls with named Gradio API endpoints.
- Updated the Python dependency set for Python 3.10, including Gradio `6.14.0`, Transformers `5.9.0`, and current compatible image-processing dependencies while preserving CUDA PyTorch installation.
- Added focused UI/process-control and terminal-logging unit tests, reproducible `requirements-dev.txt`, and stable pytest collection for optional-data integration tests.
- Added ANSI color-coded terminal logging for informational, warning, and error output, with pipe-safe and `NO_COLOR` behavior.
- Added a Pinokio app launcher with install, start, update, and reset actions.
- Updated the Pinokio launcher with dependency preflight checks, an integrated `Run Tests` action, and installation-time verification.
- Added Windows `setup.bat` and `launch.bat` entry points with an NVIDIA CUDA 12.8 PyTorch path suitable for RTX 3090 and RTX 5070 Ti systems.
- Updated the Windows setup flow to enforce Python 3.10, install test dependencies, run verification, and report missing UI dependencies before launch.
- Added Web UI, API, launcher, GPU setup, credits, and license documentation.

## Upstream History

The upstream project history remains summarized in `README.md`.
