from __future__ import annotations

import argparse
import re
import subprocess
import sys
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

STAGES = OrderedDict(
    [
        (0, ("Download", "Anime- oder Booru-Quellen herunterladen. Fuer lokale Videos nicht aktivieren.")),
        (1, ("Frames", "Frames extrahieren und aehnliche Bilder entfernen.")),
        (2, ("Crop", "Charaktere erkennen und Bildausschnitte erzeugen.")),
        (3, ("Classify", "Charakterausschnitte anhand Referenzen oder Clustern zuordnen.")),
        (4, ("Select", "Trainingsbilder auswaehlen, kopieren und skalieren.")),
        (5, ("Caption", "Tags, Captions, Core-Tags und Wildcards erzeugen.")),
        (6, ("Arrange", "Trainingsmaterial nach Konzepten und Charakteren anordnen.")),
        (7, ("Balance", "Wiederholungsgewichte fuer das Training berechnen.")),
    ]
)

FIELD_GROUPS = OrderedDict(
    [
        (
            "Allgemein",
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
            "Schritt 0 - Download",
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
            "Schritt 1 - Frames",
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
            "Schritt 2 - Crop",
            [
                "min_crop_size",
                "crop_with_head",
                "crop_with_face",
                "detect_level",
                "use_3stage_crop",
            ],
        ),
        (
            "Schritt 3 - Classify",
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
            "Schritt 4 - Select",
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
            "Schritt 5 - Caption",
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
            "Schritt 6 - Arrange",
            [
                "rearrange_up_levels",
                "arrange_format",
                "max_character_number",
                "min_images_per_combination",
            ],
        ),
        (
            "Schritt 7 - Balance",
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
  background: radial-gradient(circle at 8% 0%, #20394c 0%, #101923 34%, #090e13 100%);
  color: #f1eee6;
  font-family: "Segoe UI", sans-serif;
}
.hero {
  padding: 1.6rem 1.8rem;
  border: 1px solid #314858;
  background: linear-gradient(110deg, rgba(16, 26, 35, .92), rgba(23, 43, 55, .72));
  box-shadow: inset 4px 0 0 #f2aa4c;
  margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 .35rem; letter-spacing: .06em; text-transform: uppercase; color: #f8d59b; }
.hero p { margin: 0; color: #cad3d5; max-width: 72ch; }
.stage-grid table { width: 100%; font-size: .93rem; }
.stage-grid td:first-child { color: #f2aa4c; font-weight: 600; white-space: nowrap; }
.run-panel { border: 1px solid #334956; background: rgba(8, 14, 19, .62); padding: .8rem; }
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
    return f"Konfiguration gespeichert: {path.relative_to(ROOT)}", str(path)


def load_configuration(preset: str, uploaded_path: str | None):
    config = defaults(include_screenshots=False)
    ui_config: dict[str, Any] = {}
    path = Path(uploaded_path) if uploaded_path else ROOT / preset
    if path.exists():
        loaded = flatten_toml(toml.load(path))
        ui_config = loaded.pop("ui", {}) if isinstance(loaded.get("ui"), dict) else {}
        config.update({key: clean_value(value) for key, value in loaded.items()})
        status = f"Konfiguration geladen: {path.name}"
    else:
        status = f"Konfiguration nicht gefunden: {path}"
    selected = [str(stage) for stage in ui_config.get("enabled_stages", [3, 4, 5, 6, 7])]
    updates = [gr.update(value=selected)]
    updates.extend(gr.update(value=display_value(action, config.get(action.dest))) for action in ACTIONS)
    updates.append(status)
    return updates


def execute_config(stages: list[int], config: dict[str, Any]):
    if not stages:
        yield "Kein Schritt ausgewaehlt. Aktivieren Sie mindestens eine Checkbox."
        return

    SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG.open("w", encoding="utf-8") as handle:
        toml.dump(config, handle)

    history = [
        f"Runtime-Konfiguration: {RUNTIME_CONFIG.relative_to(ROOT)}",
        f"Schritte: {', '.join(str(stage) for stage in stages)}",
    ]
    for stage in stages:
        title = STAGES[stage][0]
        history.append(f"\n--- Schritt {stage}: {title} ---")
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
        assert process.stdout is not None
        for line in process.stdout:
            history.append(line.rstrip())
            history = history[-400:]
            yield "\n".join(history)
        return_code = process.wait()
        if return_code != 0:
            history.append(f"Schritt {stage} ist mit Exit-Code {return_code} fehlgeschlagen.")
            yield "\n".join(history)
            return
        history.append(f"Schritt {stage} abgeschlossen.")
    history.append("Alle ausgewaehlten Schritte wurden abgeschlossen.")
    yield "\n".join(history)


def run_selected_stages(selected: list[str], *values: Any):
    yield from execute_config(stage_values(selected), build_config(values, ACTIONS))


def run_saved_profile(config_path: str, stages_text: str):
    requested_path = (ROOT / config_path).resolve()
    if ROOT not in requested_path.parents or not requested_path.exists():
        yield "Die Profil-Datei muss innerhalb des Projektordners existieren."
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
    info = action.help or "Pipeline-Einstellung."
    shown = display_value(action, value)
    if isinstance(action, argparse._StoreTrueAction):
        return gr.Checkbox(label=label, value=shown, info=info)
    if action.choices:
        return gr.Dropdown(label=label, choices=list(action.choices), value=shown, info=info)
    if action.type in (int, float):
        precision = 0 if action.type is int else None
        return gr.Number(label=label, value=None if shown == "" else shown, precision=precision, info=info)
    if action.nargs in ("*", "+"):
        return gr.Textbox(label=label, value=shown, lines=2, info=f"{info} Eine Option pro Zeile oder kommasepariert.")
    return gr.Textbox(label=label, value=shown, info=info)


def build_interface() -> gr.Blocks:
    initial = defaults()
    controls: list[Any] = []
    grouped = {name: set(keys) for name, keys in FIELD_GROUPS.items()}
    known_fields = {key for keys in grouped.values() for key in keys}

    with gr.Blocks(css=STYLE, title="Anime2SD Frame Lab") as demo:
        gr.HTML(
            """
            <section class="hero">
              <h1>Anime2SD / Frame Lab</h1>
              <p>Kontrollzentrum fuer Extraktion, Klassifikation und Captioning.
              Speichern Sie Profile fuer wiederholbare Laeufe auf RTX 3090 und RTX 5070 Ti.</p>
            </section>
            """
        )
        with gr.Row():
            with gr.Column(scale=2, elem_classes="run-panel"):
                gr.Markdown("### Ablauf")
                stage_selector = gr.CheckboxGroup(
                    choices=[(f"{number} - {details[0]}", str(number)) for number, details in STAGES.items()],
                    value=["3", "4", "5", "6", "7"],
                    label="Auszufuehrende Schritte",
                    info="Schritte 1 und 2 sind optional; Schritte 3 bis 7 koennen einzeln ein- oder ausgeschaltet werden.",
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
                    run_button = gr.Button("Pipeline starten", variant="primary")
                output = gr.Textbox(label="Laufprotokoll", lines=18, interactive=False)
            with gr.Column(scale=1, elem_classes="run-panel"):
                gr.Markdown("### Konfiguration")
                preset = gr.Dropdown(
                    choices=[
                        "configs/pipelines/screenshots.toml",
                        "configs/pipelines/booru.toml",
                        "configs/pipelines/base.toml",
                    ],
                    value="configs/pipelines/screenshots.toml",
                    label="Vorlage",
                    info="Laedt eine Projektvorlage; fehlende Werte werden aus den Standardwerten ergaenzt.",
                )
                uploaded = gr.File(label="TOML-Profil importieren", file_types=[".toml"], type="filepath")
                load_button = gr.Button("Konfiguration laden")
                profile_name = gr.Textbox(label="Profilname", value="my_pipeline", info="Dateiname fuer ein gespeichertes TOML-Profil.")
                save_button = gr.Button("Konfiguration speichern")
                downloaded = gr.File(label="Gespeichertes Profil", interactive=False)
                status = gr.Markdown(f"Weboberflaeche: `http://127.0.0.1:{PORT}` (fester Port).")

        gr.Markdown("## Einstellungen pro Schritt")
        for group_name, fields in FIELD_GROUPS.items():
            with gr.Accordion(group_name, open=group_name in {"Allgemein", "Schritt 3 - Classify"}):
                group_actions = [item for item in ACTIONS if item.dest in fields]
                for offset in range(0, len(group_actions), 3):
                    with gr.Row():
                        for action in group_actions[offset:offset + 3]:
                            controls.append(make_component(action, initial.get(action.dest)))

        remaining = [action for action in ACTIONS if action.dest not in known_fields]
        if remaining:
            with gr.Accordion("Weitere Einstellungen", open=False):
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
        )
        with gr.Accordion("Programmatischer Aufruf", open=False):
            gr.Markdown(
                "Ein gespeichertes TOML-Profil kann ueber die benannte API `run_saved_profile` "
                "ausgefuehrt werden. Ohne Schrittliste gelten die im Profil gespeicherten Checkboxen."
            )
            with gr.Row():
                api_profile = gr.Textbox(label="Relativer Profilpfad", value="configs/ui/saved/my_pipeline.toml")
                api_stages = gr.Textbox(label="Schritte (optional)", value="3,4,5,6,7")
            api_button = gr.Button("Gespeichertes Profil ausfuehren")
            api_output = gr.Textbox(label="API-Laufprotokoll", lines=12, interactive=False)
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
    )
