# Changelog

## Unreleased - 2026-05-25

- Fixed Stage 3 out-of-memory failures on very large crop collections by processing CCIP/OPTICS similarity data in configurable, memory-bounded chunks (`--classification_chunk_size`, default `4096`) while preserving global saved-output handling.
- Replaced UI presets with one importable/exportable TOML configuration that stores enabled stages and stage settings together, and reduced the height of the import drop zone.
- Renamed the user-facing Stage 2 label from **Crop** to **Detect** and added `detect` as its CLI alias while retaining `crop` compatibility.
- Added the local Gradio Frame Lab interface on fixed port `7866`.
- Added independently selectable pipeline stages, per-stage settings with help text, TOML configuration save/load, and a named configuration execution API.
- Redesigned the UI with separate light/dark styling, per-stage setting tabs, and English interface text.
- Updated stage settings to show only enabled stage tabs, add stage purpose summaries, and expose concise quality, matching, and performance guidance for impactful options.
- Added Windows path normalization and documented the Stage 3 character-reference folder layout.
- Added a workspace-root workflow that creates `src`, `ref`, `logs`, `dst/intermediate`, and `dst/training` folders and applies their paths to saved and executed configurations.
- Fixed UI stage execution so consecutive selected stages share one pipeline run, preserving generated intermediate data between dependent stages.
- Replaced per-image missing-metadata warnings with one informational initialization summary and made **Stop pipeline** cancel the running UI task while terminating its process tree.
- Unified workflow editing and configuration export behind one TOML file storing stages, workspace paths, and edited settings.
- Collapsed the workflow stage explanation table into an optional **Stage guide** panel to reduce vertical space while selecting stages.
- Moved **Configuration** below **Workflow** in a two-column layout, removed the visible programmatic-access panel, mirrored pipeline progress to the terminal, and added non-destructive workspace reuse plus an explicit generated-output cleanup action.
- Made **Configuration** collapsible and automatically creates missing workspace folders at run time without altering existing content.
- Added live stage progress/completion status, condensed ANSI-free run-log progress display, and elapsed-time heartbeat reporting while OPTICS clustering is active.
- Added Stage 3 output controls for per-character and per-episode matched-image limits and optional cleanup of classified JSON/NPY auxiliary files after downstream use.
- Deferred creation of `dst` stage directories until output is produced, and changed output cleanup to leave no empty generated directory tree behind.
- Stored per-image JSON/NPY sidecars inside local `metadata` child directories throughout generated data, with preprocessing migration for older adjacent sidecars.
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
