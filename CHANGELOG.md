# Changelog

## 0.0.4 - 2026-08-20

### Added

- Added the optional Stage 2 `--keep_single_character_uncropped` mode, which preserves the complete source image when exactly one valid character is detected while retaining normal per-character crops for multi-character images.
- Added `--single_character_uncropped_min_area_ratio` to require a configurable minimum person bounding-box area before a single-character original is preserved; `0` disables the threshold.
- Added English and German Gradio controls, guidance, TOML persistence, decision metadata, and Stage 2 summary statistics for the new crop mode.
- Added focused tests for legacy behavior, zero/single/multiple detections, head/face validation, three-stage cropping, one-pass person detection, Stage 3 compatibility, and Stage 4 duplicate prevention.

### Changed

- Updated the package version to `0.0.4`.

## 0.0.3 - 2026-08-20

### Fixed

- Prevented truncated, corrupt, and unreadable image files from aborting the pipeline; confirmed invalid inputs are skipped and can be moved to quarantine.
- Hardened Stage 3 reference and dataset feature extraction without misclassifying CCIP, ONNX, CUDA, memory, or other model/runtime failures as corrupt images.
- Preserved reference-label alignment after invalid reference images are removed and added clear errors when no valid references remain.
- Kept `tqdm` output readable by using compact invalid-image warnings instead of repeated expected-error tracebacks.

### Added

- Added a central, thread-safe invalid-image handler with collision-safe quarantine paths, relative source structure preservation, move-failure handling, and per-stage counters.
- Added append-only `invalid_images.log` and `invalid_images.jsonl` records under the configured log directory.
- Added per-stage timing/invalid-image summaries and a final pipeline completion summary with per-character recognized-image counts.
- Added safe defaults and GUI controls for continuing after invalid images, quarantine behavior and location, and invalid-image logging.
- Added a browser **Reconnect live run** action plus periodic state synchronization so a reloaded page can recover the current status, progress bar, and live log while the background run continues.
- Added focused tests for valid, truncated, malformed, colliding, move-failure, non-image CCIP-error, and mixed-reference cases.
- Updated the package version to `0.0.3`.

## 0.0.2 - 2026-07-12

- Added a terminal `INFO` message when the Gradio UI configuration is saved, and color-highlighted mirrored UI terminal severity lines for `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
- Fixed Stage 2 character cropping with `--detect_level x` by selecting the available `person_detect_v0_x` model instead of the missing `person_detect_v1.1_x` model.
- Updated the package version to `0.0.2`.

## 0.0.1 - 2026-05-25

- Changed `--remove_src_files_after_pipeline` so it now clears source files, generated metadata, and source media subfolders below `src_dir` while leaving only the source root folder.
- Added separate General cleanup settings, `--remove_dst_metadata_after_pipeline` and `--remove_ref_metadata_after_pipeline`, to delete generated metadata below destination and reference folders after a successful pipeline run while keeping images.
- Added a General cleanup setting, `--remove_src_files_after_pipeline`, that deletes source files after a successful pipeline run.
- Updated README and credits to document the current Frame Lab localization, Stage 3 cleanup controls, and that the changes in this repository were implemented with Codex.
- Added an always-visible save action below stage settings and an in-page confirmation step before shutting down the UI server.
- Added English/German UI language selection for the Gradio interface, including localized stage names, page descriptions, and expanded setting explanations stored in the global configuration.
- Added a Stage 3 cleanup option to delete classified `0_noise`/`0_noisy` output after successful classification, and covered Stage 3 cleanup toggles in saved UI configuration.
- Added a Stage 3 setting to remove generated Stage 2 crop intermediates after successful classification while preserving directly supplied Stage 3 inputs.
- Fixed Stage 3 out-of-memory failures on very large crop collections by processing CCIP/OPTICS similarity data in configurable, memory-bounded chunks (`--classification_chunk_size`, default `4096`) while preserving global saved-output handling.
- Replaced UI presets with one importable/exportable TOML configuration that stores enabled stages and stage settings together, and reduced the height of the import drop zone.
- Made the TOML configuration a global auto-loaded UI setting file, removed the remaining profile-name field, and positioned server shutdown after the stage-settings area.
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
