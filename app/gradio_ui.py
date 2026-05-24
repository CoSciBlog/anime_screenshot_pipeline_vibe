from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import toml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime2sd.parse_arguments import create_parser


PORT = 7866
CONFIG_DIR = ROOT / "configs" / "ui"
SAVED_CONFIG_DIR = CONFIG_DIR / "saved"
RUNTIME_CONFIG = SAVED_CONFIG_DIR / "_last_run.toml"
PIPELINE_PROCESS_LOCK = threading.Lock()
PIPELINE_RUN_LOCK = threading.Lock()
PIPELINE_STOP_REQUESTED = threading.Event()
ACTIVE_PIPELINE_PROCESS: subprocess.Popen[str] | None = None

STAGES = OrderedDict(
    [
        (0, ("Download", "Download anime or booru sources. Disable this step for local videos.")),
        (1, ("Frames", "Extract frames and remove similar images.")),
        (2, ("Crop", "Detect characters and generate image crops.")),
        (3, ("Classify", "Assign character crops using references or clusters.")),
        (4, ("Select", "Select, copy, and resize training images.")),
        (5, ("Caption", "Generate tags, captions, core tags, and wildcards.")),
        (6, ("Arrange", "Arrange training material by concepts and characters.")),
        (7, ("Balance", "Calculate repeat weights for training.")),
    ]
)

FIELD_GROUPS = OrderedDict(
    [
        (
            "General",
            [
                "src_dir",
                "dst_dir",
                "extra_path_component",
                "log_dir",
                "log_prefix",
                "pipeline_type",
                "image_type",
                "remove_intermediate",
                "overwrite_path",
                "load_grabber_ext",
                "load_aux",
                "save_aux",
            ],
        ),
        (
            "Stage 0 - Download",
            [
                "anime_name",
                "candidate_submitters",
                "anime_resolution",
                "min_download_episode",
                "max_download_episode",
                "anime_name_booru",
                "character_info_file",
                "download_for_characters",
                "booru_download_limit",
                "booru_download_limit_per_character",
                "allowed_ratings",
                "allowed_image_classes",
                "max_download_size",
            ],
        ),
        (
            "Stage 1 - Frames",
            [
                "extract_key",
                "image_prefix",
                "ep_init",
                "no_remove_similar",
                "detect_duplicate_model",
                "detect_duplicate_batch_size",
                "similar_thresh",
            ],
        ),
        (
            "Stage 2 - Crop",
            [
                "min_crop_size",
                "crop_with_head",
                "crop_with_face",
                "detect_level",
                "use_3stage_crop",
            ],
        ),
        (
            "Stage 3 - Classify",
            [
                "character_ref_dir",
                "n_add_to_ref_per_character",
                "ignore_character_metadata",
                "no_extract_from_noise",
                "no_filter_characters",
                "keep_unnamed_clusters",
                "accept_multiple_candidates",
                "cluster_merge_threshold",
                "cluster_min_samples",
                "same_threshold_rel",
                "same_threshold_abs",
            ],
        ),
        (
            "Stage 4 - Select",
            [
                "overwrite_emb_init_info",
                "character_overwrite_uncropped",
                "character_remove_unclassified",
                "no_cropped_in_dataset",
                "no_original_in_dataset",
                "no_resize",
                "max_size",
                "image_save_ext",
                "n_anime_reg",
                "filter_again",
            ],
        ),
        (
            "Stage 5 - Caption",
            [
                "overwrite_tags",
                "tagging_method",
                "tag_threshold",
                "sort_mode",
                "append_dropped_character_tags",
                "max_tag_number",
                "blacklist_tags_file",
                "overlap_tags_file",
                "character_tags_file",
                "process_from_original_tags",
                "prune_mode",
                "drop_difficulty",
                "compute_core_tag_up_levels",
                "core_frequency_thresh",
                "use_existing_core_tag_file",
                "drop_all_core",
                "emb_min_difficulty",
                "emb_max_difficulty",
                "emb_init_all_core",
                "append_dropped_character_tags_wildcard",
                "caption_ordering",
                "caption_inner_sep",
                "caption_outer_sep",
                "character_sep",
                "character_inner_sep",
                "character_outer_sep",
                "keep_tokens_sep",
                "keep_tokens_before",
                "use_npeople_prob",
                "use_character_prob",
                "use_copyright_prob",
                "use_image_type_prob",
                "use_artist_prob",
                "use_rating_prob",
                "use_crop_info_prob",
                "use_tags_prob",
            ],
        ),
        (
            "Stage 6 - Arrange",
            [
                "rearrange_up_levels",
                "arrange_format",
                "max_character_number",
                "min_images_per_combination",
            ],
        ),
        (
            "Stage 7 - Balance",
            [
                "compute_multiply_up_levels",
                "min_multiply",
                "max_multiply",
                "weight_csv",
            ],
        ),
    ]
)

STYLE = """
.gradio-container {
  --frame-bg: #f2f4f2;
  --panel-bg: #ffffff;
  --panel-raised: #f7f8f6;
  --line: #d8ded9;
  --ink: #172127;
  --muted: #5b6b70;
  --accent: #bf5a20;
  --accent-soft: #f8e7d7;
  background:
    radial-gradient(circle at 3% 0%, #fff9ee 0, transparent 35rem),
    linear-gradient(180deg, #fafbf8 0%, var(--frame-bg) 100%);
  color: var(--ink);
  font-family: "Aptos", "Trebuchet MS", sans-serif;
  min-height: 100vh;
}
.dark .gradio-container,
.gradio-container.dark {
  --frame-bg: #0c141a;
  --panel-bg: #121d24;
  --panel-raised: #18262e;
  --line: #2b3e47;
  --ink: #f2efe9;
  --muted: #b4c1c4;
  --accent: #f2a34b;
  --accent-soft: #2b241c;
  background:
    radial-gradient(circle at 3% 0%, #20394b 0, transparent 36rem),
    linear-gradient(180deg, #0e1921 0%, var(--frame-bg) 100%);
}
.gradio-container h2,
.gradio-container h3,
.gradio-container label,
.gradio-container .prose {
  color: var(--ink);
}
.gradio-container .secondary-text,
.gradio-container .info {
  color: var(--muted);
}
.workspace {
  padding-top: .5rem;
}
.run-panel {
  border: 1px solid var(--line);
  border-radius: .7rem;
  background: var(--panel-bg);
  padding: 1rem;
  box-shadow: 0 10px 34px rgba(15, 26, 34, .06);
}
.dark .run-panel { box-shadow: 0 14px 38px rgba(0, 0, 0, .22); }
.panel-label {
  margin: 0 0 .75rem;
  color: var(--muted);
  font-size: .74rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 700;
}
.stage-grid {
  border: 1px solid var(--line);
  border-radius: .55rem;
  background: var(--panel-raised);
  overflow: hidden;
  margin: .6rem 0 .9rem;
}
.stage-grid table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.stage-grid td { padding: .62rem .7rem; border-bottom: 1px solid var(--line); color: var(--muted); }
.stage-grid tr:last-child td { border-bottom: 0; }
.stage-grid td:first-child { color: var(--accent); font-weight: 700; white-space: nowrap; }
.settings-label {
  color: var(--muted);
  font-size: .76rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin: 1.4rem 0 .65rem;
}
.settings-tabs {
  background: var(--panel-bg);
  border: 1px solid var(--line);
  border-radius: .7rem;
  padding: .55rem;
  box-shadow: 0 10px 34px rgba(15, 26, 34, .05);
}
.settings-tabs button {
  color: var(--muted);
  font-weight: 600;
}
.settings-tabs button.selected {
  color: var(--accent);
  border-color: var(--accent);
}
.settings-grid {
  padding-top: .7rem;
}
.gradio-container .primary {
  background: var(--accent);
  border-color: var(--accent);
}
.shutdown-button {
  margin-top: .8rem;
}
.gradio-container a { color: var(--accent); }
"""


def parser_actions() -> list[argparse.Action]:
    skipped = {"help", "base_config_file", "config_file", "start_stage", "end_stage"}
    return [
        action
        for action in create_parser()._actions
        if action.dest not in skipped
    ]


def flatten_toml(data: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if key == "ui":
            flattened[key] = value
        elif isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def clean_value(value: Any) -> Any:
    return None if value == {} else value


def defaults(include_screenshots: bool = True) -> dict[str, Any]:
    values = {action.dest: clean_value(action.default) for action in parser_actions()}
    paths = [ROOT / "configs" / "pipelines" / "base.toml"]
    if include_screenshots:
        paths.append(ROOT / "configs" / "pipelines" / "screenshots.toml")
    for path in paths:
        if path.exists():
            values.update(
                {
                    key: clean_value(value)
                    for key, value in flatten_toml(toml.load(path)).items()
                    if key != "ui"
                }
            )
    return values


def text_for_list(value: Any) -> str:
    if not value:
        return ""
    return "\n".join(str(item) for item in value)


def display_value(action: argparse.Action, value: Any) -> Any:
    value = clean_value(value)
    if action.nargs in ("*", "+"):
        return text_for_list(value)
    if isinstance(action, argparse._StoreTrueAction):
        return bool(value)
    return "" if value is None else value


def component_value(action: argparse.Action, value: Any) -> Any:
    shown = display_value(action, value)
    return None if action.choices and shown == "" else shown


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,]+", value or "") if item.strip()]


def config_value(action: argparse.Action, value: Any) -> Any:
    if action.nargs in ("*", "+"):
        return parse_list(str(value))
    if isinstance(action, argparse._StoreTrueAction):
        return bool(value)
    if value == "" or value is None:
        return None
    if action.type in (int, float):
        return action.type(value)
    return value


def build_config(values: Iterable[Any], actions: list[argparse.Action]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for action, value in zip(actions, values):
        converted = config_value(action, value)
        if converted is not None:
            config[action.dest] = converted
    return config


def stage_values(selected: Iterable[Any] | None) -> list[int]:
    return sorted({int(stage) for stage in (selected or [])})


def profile_path(name: str) -> Path:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "pipeline_profile"
    return SAVED_CONFIG_DIR / f"{cleaned}.toml"


def save_configuration(name: str, selected: list[str], *values: Any):
    SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config(values, ACTIONS)
    config["ui"] = {"enabled_stages": stage_values(selected), "fixed_port": PORT}
    path = profile_path(name)
    with path.open("w", encoding="utf-8") as handle:
        toml.dump(config, handle)
    return f"Configuration saved: {path.relative_to(ROOT)}", str(path)


def load_configuration(preset: str, uploaded_path: str | None):
    config = defaults(include_screenshots=False)
    ui_config: dict[str, Any] = {}
    path = Path(uploaded_path) if uploaded_path else ROOT / preset
    if path.exists():
        loaded = flatten_toml(toml.load(path))
        ui_config = loaded.pop("ui", {}) if isinstance(loaded.get("ui"), dict) else {}
        config.update({key: clean_value(value) for key, value in loaded.items()})
        status = f"Configuration loaded: {path.name}"
    else:
        status = f"Configuration not found: {path}"
    selected = [str(stage) for stage in ui_config.get("enabled_stages", [3, 4, 5, 6, 7])]
    updates = [gr.update(value=selected)]
    updates.extend(gr.update(value=component_value(action, config.get(action.dest))) for action in ACTIONS)
    updates.append(status)
    return updates


def stop_pipeline() -> str:
    global ACTIVE_PIPELINE_PROCESS
    with PIPELINE_PROCESS_LOCK:
        process = ACTIVE_PIPELINE_PROCESS
        if process is None or process.poll() is not None:
            return "No pipeline process is currently running."
        PIPELINE_STOP_REQUESTED.set()
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    return "Stopping the active pipeline..."


def shutdown_server() -> str:
    stop_pipeline()

    def exit_process() -> None:
        time.sleep(0.35)
        os._exit(0)

    threading.Thread(target=exit_process, daemon=True).start()
    return "Server shutdown requested. This browser connection will close shortly."


def execute_config(stages: list[int], config: dict[str, Any]):
    global ACTIVE_PIPELINE_PROCESS
    if not stages:
        yield "No stage selected. Enable at least one checkbox."
        return
    if not PIPELINE_RUN_LOCK.acquire(blocking=False):
        yield "A pipeline run is already active. Stop it before starting another run."
        return

    PIPELINE_STOP_REQUESTED.clear()
    try:
        SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG.open("w", encoding="utf-8") as handle:
            toml.dump(config, handle)

        history = [
            f"Runtime configuration: {RUNTIME_CONFIG.relative_to(ROOT)}",
            f"Stages: {', '.join(str(stage) for stage in stages)}",
        ]
        for stage in stages:
            if PIPELINE_STOP_REQUESTED.is_set():
                history.append("Pipeline stopped by user.")
                yield "\n".join(history)
                return
            title = STAGES[stage][0]
            history.append(f"\n--- Stage {stage}: {title} ---")
            yield "\n".join(history)
            command = [
                sys.executable,
                str(ROOT / "automatic_pipeline.py"),
                "--base_config_file",
                str(RUNTIME_CONFIG),
                "--start_stage",
                str(stage),
                "--end_stage",
                str(stage),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with PIPELINE_PROCESS_LOCK:
                ACTIVE_PIPELINE_PROCESS = process
            assert process.stdout is not None
            for line in process.stdout:
                history.append(line.rstrip())
                history = history[-400:]
                yield "\n".join(history)
            return_code = process.wait()
            with PIPELINE_PROCESS_LOCK:
                if ACTIVE_PIPELINE_PROCESS is process:
                    ACTIVE_PIPELINE_PROCESS = None
            if PIPELINE_STOP_REQUESTED.is_set():
                history.append("Pipeline stopped by user.")
                yield "\n".join(history)
                return
            if return_code != 0:
                history.append(f"Stage {stage} failed with exit code {return_code}.")
                yield "\n".join(history)
                return
            history.append(f"Stage {stage} completed.")
        history.append("All selected stages completed.")
        yield "\n".join(history)
    finally:
        with PIPELINE_PROCESS_LOCK:
            process = ACTIVE_PIPELINE_PROCESS
            ACTIVE_PIPELINE_PROCESS = None
        if process is not None and process.poll() is None:
            process.terminate()
        PIPELINE_STOP_REQUESTED.clear()
        PIPELINE_RUN_LOCK.release()


def run_selected_stages(selected: list[str], *values: Any):
    yield from execute_config(stage_values(selected), build_config(values, ACTIONS))


def run_saved_profile(config_path: str, stages_text: str):
    requested_path = (ROOT / config_path).resolve()
    if ROOT not in requested_path.parents or not requested_path.exists():
        yield "The profile file must exist inside the project directory."
        return
    loaded = flatten_toml(toml.load(requested_path))
    ui_config = loaded.pop("ui", {}) if isinstance(loaded.get("ui"), dict) else {}
    if stages_text.strip():
        stages = stage_values(parse_list(stages_text))
    else:
        stages = stage_values(ui_config.get("enabled_stages", []))
    yield from execute_config(stages, loaded)


def make_component(action: argparse.Action, value: Any):
    label = f"--{action.dest}"
    info = action.help or "Pipeline setting."
    shown = component_value(action, value)
    if isinstance(action, argparse._StoreTrueAction):
        return gr.Checkbox(label=label, value=shown, info=info)
    if action.choices:
        return gr.Dropdown(label=label, choices=list(action.choices), value=shown, info=info)
    if action.type in (int, float):
        precision = 0 if action.type is int else None
        return gr.Number(label=label, value=None if shown == "" else shown, precision=precision, info=info)
    if action.nargs in ("*", "+"):
        return gr.Textbox(label=label, value=shown, lines=2, info=f"{info} Enter one option per line or separate values with commas.")
    return gr.Textbox(label=label, value=shown, info=info)


def build_interface() -> gr.Blocks:
    initial = defaults()
    controls: list[Any] = []
    grouped = {name: set(keys) for name, keys in FIELD_GROUPS.items()}
    known_fields = {key for keys in grouped.values() for key in keys}

    with gr.Blocks(
        title="Anime2SD Frame Lab",
        elem_classes="workspace",
    ) as demo:
        with gr.Row():
            with gr.Column(scale=2, elem_classes="run-panel"):
                gr.HTML("<p class='panel-label'>Workflow</p>")
                stage_selector = gr.CheckboxGroup(
                    choices=[(f"{number} - {details[0]}", str(number)) for number, details in STAGES.items()],
                    value=["3", "4", "5", "6", "7"],
                    label="Stages to run",
                    info="Stages 1 and 2 are optional; stages 3 through 7 can be enabled or disabled individually.",
                )
                gr.HTML(
                    "<div class='stage-grid'><table>"
                    + "".join(
                        f"<tr><td>{number} - {title}</td><td>{description}</td></tr>"
                        for number, (title, description) in STAGES.items()
                    )
                    + "</table></div>"
                )
                with gr.Row():
                    run_button = gr.Button("Run pipeline", variant="primary")
                    stop_button = gr.Button("Stop pipeline", variant="stop")
                output = gr.Textbox(label="Run log", lines=18, interactive=False)
            with gr.Column(scale=1, elem_classes="run-panel"):
                gr.HTML("<p class='panel-label'>Configuration</p>")
                preset = gr.Dropdown(
                    choices=[
                        "configs/pipelines/screenshots.toml",
                        "configs/pipelines/booru.toml",
                        "configs/pipelines/base.toml",
                    ],
                    value="configs/pipelines/screenshots.toml",
                    label="Preset",
                    info="Load a project preset; missing values are filled from defaults.",
                )
                uploaded = gr.File(label="Import TOML profile", file_types=[".toml"], type="filepath")
                load_button = gr.Button("Load configuration")
                profile_name = gr.Textbox(label="Profile name", value="my_pipeline", info="File name for a saved TOML profile.")
                save_button = gr.Button("Save configuration")
                downloaded = gr.File(label="Saved profile", interactive=False)
                status = gr.Markdown(f"Web interface: `http://127.0.0.1:{PORT}` (fixed port).")
                shutdown_button = gr.Button("Shut down server", variant="stop", elem_classes="shutdown-button")

        gr.HTML("<p class='settings-label'>Settings by stage</p>")
        with gr.Tabs(elem_classes="settings-tabs"):
            for group_name, fields in FIELD_GROUPS.items():
                with gr.Tab(group_name):
                    group_actions = [item for item in ACTIONS if item.dest in fields]
                    with gr.Column(elem_classes="settings-grid"):
                        for offset in range(0, len(group_actions), 3):
                            with gr.Row():
                                for action in group_actions[offset:offset + 3]:
                                    controls.append(make_component(action, initial.get(action.dest)))

            remaining = [action for action in ACTIONS if action.dest not in known_fields]
            if remaining:
                with gr.Tab("Additional settings"):
                    with gr.Column(elem_classes="settings-grid"):
                        for offset in range(0, len(remaining), 3):
                            with gr.Row():
                                for action in remaining[offset:offset + 3]:
                                    controls.append(make_component(action, initial.get(action.dest)))

        ordered_controls = {action.dest: control for action, control in zip(
            [item for group in FIELD_GROUPS.values() for item in ACTIONS if item.dest in group] + remaining,
            controls,
        )}
        controls_in_action_order = [ordered_controls[action.dest] for action in ACTIONS]

        save_button.click(
            save_configuration,
            inputs=[profile_name, stage_selector, *controls_in_action_order],
            outputs=[status, downloaded],
            api_name="save_configuration",
        )
        load_button.click(
            load_configuration,
            inputs=[preset, uploaded],
            outputs=[stage_selector, *controls_in_action_order, status],
        )
        run_button.click(
            run_selected_stages,
            inputs=[stage_selector, *controls_in_action_order],
            outputs=output,
            api_name="run_from_form",
            concurrency_limit=1,
            concurrency_id="pipeline_run",
        )
        stop_button.click(
            stop_pipeline,
            outputs=output,
            api_name="stop_pipeline",
            concurrency_limit=None,
        )
        shutdown_button.click(
            shutdown_server,
            outputs=status,
            api_name="shutdown_server",
            concurrency_limit=None,
        )
        with gr.Accordion("Programmatic access", open=False):
            gr.Markdown(
                "A saved TOML profile can be executed through the named `run_saved_profile` API. "
                "When no stage list is provided, the checkboxes stored in the profile are used."
            )
            with gr.Row():
                api_profile = gr.Textbox(label="Relative profile path", value="configs/ui/saved/my_pipeline.toml")
                api_stages = gr.Textbox(label="Stages (optional)", value="3,4,5,6,7")
            api_button = gr.Button("Run saved profile")
            api_output = gr.Textbox(label="API run log", lines=12, interactive=False)
            api_button.click(
                run_saved_profile,
                inputs=[api_profile, api_stages],
                outputs=api_output,
                api_name="run_saved_profile",
            )
    return demo


ACTIONS = parser_actions()


if __name__ == "__main__":
    build_interface().queue().launch(
        server_name="127.0.0.1",
        server_port=PORT,
        show_error=True,
        css=STYLE,
    )
