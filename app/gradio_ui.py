from __future__ import annotations

import argparse
import html
import os
import queue
import re
import signal
import shutil
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
GLOBAL_CONFIG = CONFIG_DIR / "configuration.toml"
RUNTIME_CONFIG = SAVED_CONFIG_DIR / "_last_run.toml"
PIPELINE_PROCESS_LOCK = threading.Lock()
PIPELINE_RUN_LOCK = threading.Lock()
PIPELINE_STOP_REQUESTED = threading.Event()
ACTIVE_PIPELINE_PROCESS: subprocess.Popen[str] | None = None
WORKSPACE_DIRECTORIES = ("src", "ref", "logs")
WORKSPACE_RESERVED_DIRECTORIES = (*WORKSPACE_DIRECTORIES, "dst")
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
STAGE_START_RE = re.compile(r"Start stage\s+(\d+)")
TQDM_PROGRESS_RE = re.compile(
    r"(?P<label>[^:\n]+):\s*(?P<percent>\d+)%\|.*?\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<timing>[^\]]+)\]"
)

STAGES = OrderedDict(
    [
        (0, ("Download", "Download anime or booru sources. Affects source variety, size, and download time.")),
        (1, ("Frames", "Extract frames and remove near duplicates. Reduces repetition before later analysis.")),
        (2, ("Detect", "Detect characters and create crops. Detection choices trade recall against runtime.")),
        (3, ("Classify", "Match crops to reference characters or clusters. This is where reference images are used.")),
        (4, ("Select", "Build the training image set and resize exports. Controls dataset quality and disk size.")),
        (5, ("Caption", "Generate tags and captions. Thresholds change caption precision and training signal.")),
        (6, ("Arrange", "Organize images by concepts and characters for readable training subsets.")),
        (7, ("Balance", "Calculate repeat weights. Changes how strongly subsets contribute during training.")),
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
            "Stage 2 - Detect",
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
                "max_images_per_character",
                "max_images_per_character_per_episode",
                "remove_classified_aux_files",
                "remove_stage2_crops_after_classification",
                "ignore_character_metadata",
                "no_extract_from_noise",
                "no_filter_characters",
                "keep_unnamed_clusters",
                "accept_multiple_candidates",
                "cluster_merge_threshold",
                "cluster_min_samples",
                "classification_chunk_size",
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

DEFAULT_ENABLED_STAGES = ["3", "4", "5", "6", "7"]
GROUP_STAGES = {
    f"Stage {number} - {title}": number
    for number, (title, _description) in STAGES.items()
}
PATH_FIELDS = {
    "src_dir",
    "dst_dir",
    "log_dir",
    "character_info_file",
    "character_ref_dir",
    "blacklist_tags_file",
    "overlap_tags_file",
    "character_tags_file",
    "weight_csv",
}
FIELD_GUIDANCE = {
    "src_dir": (
        "Input media for the first selected stage; do not use this for reference images. "
        r"The workspace button maps this to <root>\src."
    ),
    "dst_dir": r"Output root. The workspace button maps this to <root>\dst, which contains intermediate and training output.",
    "character_ref_dir": (
        "Reference image root used only in Stage 3. Create one subfolder per character, "
        r"for example <root>\ref\frieren\*.png."
    ),
    "candidate_submitters": "Narrower sources may improve consistency but can reduce available episodes.",
    "anime_resolution": "Higher resolution can preserve detail but increases download size and later processing cost.",
    "booru_download_limit": "More images increase coverage and download/runtime cost; low limits can miss rare poses.",
    "booru_download_limit_per_character": "Higher limits improve per-character coverage at added download and filtering cost.",
    "allowed_ratings": "Filtering ratings changes content distribution and may reduce usable image count.",
    "allowed_image_classes": "Stricter classes improve dataset consistency but reduce variety.",
    "max_download_size": "Smaller values reduce disk/runtime cost but discard fine detail during resizing.",
    "extract_key": "Key frames run faster and reduce duplicates, but may lose useful poses.",
    "no_remove_similar": "Keeping similar frames increases dataset size and repetition.",
    "detect_duplicate_model": "Larger or stronger models may improve duplicate matching at greater runtime cost.",
    "detect_duplicate_batch_size": "Larger batches can improve throughput but require more VRAM/RAM.",
    "similar_thresh": "Higher values remove fewer near-duplicates; lower values remove more variety.",
    "min_crop_size": "Higher values reject small/low-detail crops but yield fewer samples.",
    "crop_with_head": "Requiring a head improves identity evidence but drops valid body-only crops.",
    "crop_with_face": "Requiring a face strengthens identity matching but reduces recall.",
    "detect_level": "Higher-capacity detection can improve crop recall with slower processing.",
    "use_3stage_crop": "Additional head/halfbody crops can improve training coverage but are slow to generate.",
    "n_add_to_ref_per_character": "Expands references after matching; may improve later runs but can propagate mistakes.",
    "max_images_per_character": "Caps recognized samples per reference character; reduces imbalance and storage at the cost of pose variety.",
    "max_images_per_character_per_episode": "Caps repeated episode-specific matches per character to reduce near-duplicate dominance.",
    "remove_classified_aux_files": "Saves disk space after classification output is consumed, but prevents later reuse of cached features and metadata.",
    "remove_stage2_crops_after_classification": (
        "Saves disk space after successful classification by removing generated Stage 2 crops. "
        "User-provided Stage 3 input is never deleted."
    ),
    "no_filter_characters": "Disabling consistency filtering retains more samples at higher label-noise risk.",
    "keep_unnamed_clusters": "Keeps unmatched material for coverage, but it does not gain reference labels.",
    "cluster_merge_threshold": "Controls cluster joining; permissive matching risks merging different characters.",
    "cluster_min_samples": "Higher values suppress small clusters but can lose rare-character samples.",
    "classification_chunk_size": (
        "Bounds quadratic Stage 3 similarity memory. Lower values use less RAM/VRAM "
        "but do not merge unnamed clusters across chunks."
    ),
    "same_threshold_rel": "Changes noise extraction and filtering strictness, affecting character-match precision.",
    "same_threshold_abs": "Changes noise extraction and filtering strictness for larger clusters.",
    "no_cropped_in_dataset": "Excluding crops reduces close-up examples and dataset size.",
    "no_original_in_dataset": "Excluding originals reduces scene/context coverage and dataset size.",
    "no_resize": "Preserves native detail but increases storage and training preprocessing cost.",
    "max_size": "Larger exports preserve detail but increase disk use and training cost.",
    "filter_again": "Runs duplicate filtering again for cleaner results at extra processing time.",
    "tagging_method": "Model choice affects tag accuracy and inference speed.",
    "tag_threshold": "Higher thresholds improve tag precision but reduce descriptive coverage.",
    "max_tag_number": "More tags add detail but can dilute important training concepts.",
    "prune_mode": "Pruning removes redundant tags and changes the strength of retained concepts.",
    "core_frequency_thresh": "Higher values mark fewer tags as core, preserving more caption detail.",
    "use_character_prob": "Lower values reduce identity conditioning in captions.",
    "use_tags_prob": "Lower values reduce descriptive conditioning in captions.",
    "arrange_format": "Folder grouping affects how concepts are separated for balancing and training.",
    "min_images_per_combination": "Higher values merge sparse concepts, improving stability but losing granularity.",
    "min_multiply": "Sets the minimum training exposure for underrepresented groups.",
    "max_multiply": "Caps oversampling; lower caps reduce overfitting risk for small groups.",
    "weight_csv": "Custom weights directly alter subset exposure during training.",
}

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
.workspace-card {
  margin: .2rem 0 .85rem;
  padding: .8rem;
  border-radius: .55rem;
  border: 1px solid var(--line);
  background: var(--panel-raised);
}
.workspace-card .prose {
  color: var(--muted);
  font-size: .89rem;
}
.pipeline-status {
  border: 1px solid var(--line);
  border-radius: .55rem;
  background: var(--panel-raised);
  margin: .85rem 0 .5rem;
  padding: .75rem .85rem;
}
.pipeline-status h3 {
  color: var(--ink);
  font-size: 1rem;
  margin: 0 0 .2rem;
}
.pipeline-status p {
  color: var(--muted);
  margin: 0;
}
.pipeline-progress {
  border: 1px solid var(--line);
  border-radius: .55rem;
  background: var(--panel-raised);
  margin-bottom: .7rem;
  padding: .65rem .8rem;
}
.pipeline-progress-header {
  color: var(--muted);
  display: flex;
  font-size: .85rem;
  justify-content: space-between;
  margin-bottom: .45rem;
}
.pipeline-progress-track {
  background: var(--line);
  border-radius: 999px;
  height: .55rem;
  overflow: hidden;
}
.pipeline-progress-fill {
  background: var(--accent);
  border-radius: 999px;
  height: 100%;
  transition: width .22s ease;
}
.config-accordion {
  margin-top: 1rem;
}
.compact-upload {
  min-height: 6.2rem !important;
  max-height: 7.8rem !important;
}
.compact-upload .upload-container,
.compact-upload [data-testid="dropzone"] {
  min-height: 5rem !important;
  padding: .35rem !important;
}
.compact-upload .upload-container svg,
.compact-upload [data-testid="dropzone"] svg {
  width: 1.2rem !important;
  height: 1.2rem !important;
}
.run-log textarea {
  font-family: "Cascadia Mono", "Consolas", monospace;
  line-height: 1.42;
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
    if action.type in (int, float):
        return None if shown == "" else action.type(shown)
    return None if action.choices and shown == "" else shown


def normalize_path_value(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    return os.path.normpath(expanded) if expanded else ""


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,]+", value or "") if item.strip()]


def config_value(action: argparse.Action, value: Any) -> Any:
    if action.nargs in ("*", "+"):
        return parse_list(str(value))
    if isinstance(action, argparse._StoreTrueAction):
        return bool(value)
    if value == "" or value is None:
        return None
    if action.dest in PATH_FIELDS:
        return normalize_path_value(str(value))
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


def workspace_paths(root_value: str) -> dict[str, str]:
    root_path = normalize_path_value(root_value)
    if not root_path:
        return {}
    root = Path(root_path)
    return {
        "src_dir": str(root / "src"),
        "dst_dir": str(root / "dst"),
        "character_ref_dir": str(root / "ref"),
        "log_dir": str(root / "logs"),
    }


def apply_workspace_paths(root_value: str, config: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(config)
    mapped.update(workspace_paths(root_value))
    return mapped


def create_workspace_structure(root_value: str):
    paths = workspace_paths(root_value)
    if not paths:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Enter a workspace root before creating folders.",
        )
    root = Path(normalize_path_value(root_value))
    existing_directories = [
        relative_path for relative_path in WORKSPACE_RESERVED_DIRECTORIES
        if (root / relative_path).is_dir()
    ]
    for relative_path in WORKSPACE_RESERVED_DIRECTORIES:
        path = root / relative_path
        if path.exists() and not path.is_dir():
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                f"Cannot create workspace: `{path}` exists but is not a folder.",
            )
    for relative_path in WORKSPACE_DIRECTORIES:
        path = root / relative_path
        path.mkdir(parents=True, exist_ok=True)
    if existing_directories:
        status = (
            f"Workspace ready: `{root}`. Existing folders and their contents were kept; "
            "missing input, reference, and log folders were created. Output folders "
            "are created only when a stage writes data."
        )
    else:
        status = (
            f"Workspace created: `{root}`. Place first-stage input in `src`, "
            "character references in `ref`, and outputs will be written under `dst`."
        )
    return (
        gr.update(value=paths["src_dir"]),
        gr.update(value=paths["dst_dir"]),
        gr.update(value=paths["character_ref_dir"]),
        gr.update(value=paths["log_dir"]),
        status,
    )


def clear_workspace_output(root_value: str) -> str:
    paths = workspace_paths(root_value)
    if not paths:
        return "Enter a workspace root before clearing generated output."
    with PIPELINE_PROCESS_LOCK:
        process = ACTIVE_PIPELINE_PROCESS
        if process is not None and process.poll() is None:
            return "Stop the active pipeline before clearing generated output."
    root = Path(normalize_path_value(root_value))
    dst = Path(paths["dst_dir"])
    if dst.resolve().parent != root.resolve():
        return "Output cleanup stopped because the destination is outside the workspace root."
    if dst.exists() and not dst.is_dir():
        return f"Cannot clear generated output: `{dst}` is not a folder."
    removed = 0
    if dst.exists():
        for item in dst.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
    if removed:
        shutil.rmtree(dst)
        return (
            f"Generated output cleared: `{dst}`. Output folders will be created "
            "when a stage writes data."
        )
    if dst.exists():
        dst.rmdir()
    return (
        f"No generated output to clear in `{dst}`. Output folders will be created "
        "when a stage writes data."
    )


def clean_run_line(line: str) -> str:
    clean_fragments = (
        ANSI_ESCAPE_RE.sub("", line).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    )
    return clean_fragments[-1].strip() if clean_fragments else ""


def append_run_history(history: list[str], line: str) -> None:
    progress_match = TQDM_PROGRESS_RE.search(line)
    if progress_match:
        label = progress_match.group("label").strip()
        for index in range(len(history) - 1, max(-1, len(history) - 10), -1):
            if history[index].startswith(f"{label}:"):
                history[index] = line
                return
    history.append(line)
    del history[:-160]


def run_detail_from_line(line: str) -> str | None:
    match = TQDM_PROGRESS_RE.search(line)
    if not match:
        return None
    return (
        f"{match.group('label').strip()}: {match.group('percent')}% "
        f"({match.group('done')}/{match.group('total')}) [{match.group('timing')}]"
    )


def status_markup(title: str, message: str) -> str:
    return (
        "<div class='pipeline-status'>"
        f"<h3>{html.escape(title)}</h3><p>{html.escape(message)}</p>"
        "</div>"
    )


def progress_markup(
    selected: list[int],
    completed: set[int],
    current: int | None,
    detail: str = "",
) -> str:
    total = len(selected)
    complete_count = len(completed.intersection(selected))
    percent = round((complete_count / total) * 100) if total else 0
    current_text = "Waiting to start"
    if current is not None:
        current_text = f"Current: Stage {current} - {STAGES[current][0]}"
    elif total and complete_count == total:
        current_text = "All selected stages complete"
    label = f"{complete_count} of {total} stages complete"
    if detail:
        current_text = f"{current_text} | {detail}"
    return (
        "<div class='pipeline-progress'>"
        "<div class='pipeline-progress-header'>"
        f"<span>{html.escape(current_text)}</span><span>{html.escape(label)}</span>"
        "</div>"
        "<div class='pipeline-progress-track'>"
        f"<div class='pipeline-progress-fill' style='width:{percent}%'></div>"
        "</div></div>"
    )


def mirror_run_output(line: str) -> None:
    print(line, end="" if line.endswith("\n") else "\n", flush=True)


def process_output_with_heartbeat(process: subprocess.Popen[str]):
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    while True:
        try:
            message = messages.get(timeout=1)
        except queue.Empty:
            yield ""
            continue
        if message is None:
            return
        yield message


def stage_values(selected: Iterable[Any] | None) -> list[int]:
    return sorted({int(stage) for stage in (selected or [])})


def stage_ranges(stages: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for stage in sorted(stages):
        if ranges and stage == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], stage)
        else:
            ranges.append((stage, stage))
    return ranges


def stage_tab_updates(selected: Iterable[Any] | None):
    enabled = set(stage_values(selected))
    return [gr.update(visible=number in enabled) for number in STAGES]


def save_configuration(
    workspace_root: str,
    selected: list[str],
    *values: Any,
):
    GLOBAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config = apply_workspace_paths(workspace_root, build_config(values, ACTIONS))
    config["ui"] = {
        "enabled_stages": stage_values(selected),
        "fixed_port": PORT,
        "workspace_root": normalize_path_value(workspace_root),
    }
    with GLOBAL_CONFIG.open("w", encoding="utf-8") as handle:
        toml.dump(config, handle)
    relative_path = str(GLOBAL_CONFIG.relative_to(ROOT)).replace("\\", "/")
    return (
        f"Global configuration saved. Export is ready: `{relative_path}`",
        str(GLOBAL_CONFIG),
    )


def configuration_state(uploaded_path: str | None = None):
    config = defaults()
    ui_config: dict[str, Any] = {}
    path = Path(uploaded_path) if uploaded_path else GLOBAL_CONFIG
    if path and path.exists():
        loaded = flatten_toml(toml.load(path))
        ui_config = loaded.pop("ui", {}) if isinstance(loaded.get("ui"), dict) else {}
        config.update({key: clean_value(value) for key, value in loaded.items()})
        source = "Imported configuration" if uploaded_path else "Global configuration"
        status = f"{source} loaded: `{path.name}`"
    elif uploaded_path:
        status = "Choose a valid TOML configuration file to import."
    else:
        status = "No global configuration saved yet. Adjust settings and select Save configuration."
    selected = [str(stage) for stage in ui_config.get("enabled_stages", DEFAULT_ENABLED_STAGES)]
    workspace_root = ui_config.get("workspace_root", "")
    return config, selected, workspace_root, status


def load_configuration(uploaded_path: str | None = None):
    config, selected, workspace_root, status = configuration_state(uploaded_path)
    updates = [
        gr.update(value=workspace_root),
        gr.update(value=selected),
    ]
    updates.extend(gr.update(value=component_value(action, config.get(action.dest))) for action in ACTIONS)
    updates.append(status)
    return updates


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            text=True,
        )
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def stop_pipeline() -> str:
    global ACTIVE_PIPELINE_PROCESS
    with PIPELINE_PROCESS_LOCK:
        process = ACTIVE_PIPELINE_PROCESS
        if process is None or process.poll() is not None:
            return "No pipeline process is currently running."
        PIPELINE_STOP_REQUESTED.set()
        terminate_process_tree(process)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    return "Pipeline stopped."


def shutdown_server() -> str:
    stop_pipeline()

    def exit_process() -> None:
        time.sleep(0.35)
        os._exit(0)

    threading.Thread(target=exit_process, daemon=True).start()
    return "Server shutdown requested. This browser connection will close shortly."


def execute_config(stages: list[int], config: dict[str, Any]):
    global ACTIVE_PIPELINE_PROCESS
    completed: set[int] = set()
    current_stage: int | None = None
    detail = ""
    clustering_started: float | None = None

    def update(title: str, message: str, history: list[str]):
        return (
            status_markup(title, message),
            progress_markup(stages, completed, current_stage, detail),
            "\n".join(history),
        )

    if not stages:
        yield (
            status_markup("No stages selected", "Enable at least one stage before running."),
            progress_markup([], set(), None),
            "No stage selected. Enable at least one checkbox.",
        )
        return
    if not PIPELINE_RUN_LOCK.acquire(blocking=False):
        yield (
            status_markup("Pipeline already running", "Stop the active run before starting another."),
            progress_markup(stages, set(), None),
            "A pipeline run is already active. Stop it before starting another run.",
        )
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
        yield update("Pipeline starting", f"Preparing {len(stages)} selected stage(s).", history)
        for start_stage, end_stage in stage_ranges(stages):
            if PIPELINE_STOP_REQUESTED.is_set():
                append_run_history(history, "Pipeline stopped by user.")
                yield update("Pipeline stopped", "Processing was stopped by the user.", history)
                return
            titles = ", ".join(
                f"{stage}: {STAGES[stage][0]}" for stage in range(start_stage, end_stage + 1)
            )
            append_run_history(history, f"--- Stages {titles} ---")
            yield update("Pipeline running", f"Starting {titles}.", history)
            command = [
                sys.executable,
                str(ROOT / "automatic_pipeline.py"),
                "--base_config_file",
                str(RUNTIME_CONFIG),
                "--start_stage",
                str(start_stage),
                "--end_stage",
                str(end_stage),
            ]
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_options["start_new_session"] = True
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **popen_options,
            )
            with PIPELINE_PROCESS_LOCK:
                ACTIVE_PIPELINE_PROCESS = process
            for raw_line in process_output_with_heartbeat(process):
                if not raw_line:
                    if clustering_started is not None:
                        elapsed = int(time.monotonic() - clustering_started)
                        detail = (
                            f"Clustering active; elapsed {elapsed}s; "
                            "ETA unavailable during OPTICS fitting."
                        )
                        yield update(
                            "Pipeline running",
                            "Classification clustering is still processing.",
                            history,
                        )
                    continue
                mirror_run_output(raw_line)
                line = clean_run_line(raw_line)
                if not line:
                    continue
                stage_match = STAGE_START_RE.search(line)
                if stage_match:
                    next_stage = int(stage_match.group(1))
                    if current_stage is not None and current_stage != next_stage:
                        completed.add(current_stage)
                    current_stage = next_stage
                    detail = ""
                    clustering_started = None
                parsed_detail = run_detail_from_line(line)
                if parsed_detail:
                    detail = parsed_detail
                if "Clustering OPTICS:" in line:
                    clustering_started = time.monotonic()
                    detail = "Clustering active; ETA unavailable during OPTICS fitting."
                append_run_history(history, line)
                running_stage = current_stage if current_stage is not None else start_stage
                yield update(
                    "Pipeline running",
                    f"Processing Stage {running_stage} - {STAGES[running_stage][0]}.",
                    history,
                )
            return_code = process.wait()
            with PIPELINE_PROCESS_LOCK:
                if ACTIVE_PIPELINE_PROCESS is process:
                    ACTIVE_PIPELINE_PROCESS = None
            if PIPELINE_STOP_REQUESTED.is_set():
                append_run_history(history, "Pipeline stopped by user.")
                yield update("Pipeline stopped", "Processing was stopped by the user.", history)
                return
            if return_code != 0:
                append_run_history(
                    history,
                    f"Stages {start_stage}-{end_stage} failed with exit code {return_code}."
                )
                yield update("Pipeline failed", "Review the run log for the reported error.", history)
                return
            completed.update(stage for stage in stages if start_stage <= stage <= end_stage)
            current_stage = None
            detail = ""
            clustering_started = None
            append_run_history(history, f"Stages {start_stage}-{end_stage} completed.")
            yield update("Pipeline running", "Continuing with remaining selected stages.", history)
        append_run_history(history, "All selected stages completed.")
        yield update(
            "Pipeline complete",
            f"Completed {len(stages)} selected stage(s).",
            history,
        )
    finally:
        with PIPELINE_PROCESS_LOCK:
            process = ACTIVE_PIPELINE_PROCESS
            ACTIVE_PIPELINE_PROCESS = None
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
        PIPELINE_STOP_REQUESTED.clear()
        PIPELINE_RUN_LOCK.release()


def run_selected_stages(selected: list[str], workspace_root: str, *values: Any):
    stages = stage_values(selected)
    if normalize_path_value(workspace_root):
        *_updates, workspace_status = create_workspace_structure(workspace_root)
        if workspace_status.startswith("Cannot create workspace"):
            yield (
                status_markup("Workspace unavailable", workspace_status),
                progress_markup(stages, set(), None),
                workspace_status,
            )
            return
    config = apply_workspace_paths(workspace_root, build_config(values, ACTIONS))
    yield from execute_config(stages, config)


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
    workspace_root = ui_config.get("workspace_root", "")
    if normalize_path_value(workspace_root):
        *_updates, workspace_status = create_workspace_structure(str(workspace_root))
        if workspace_status.startswith("Cannot create workspace"):
            yield workspace_status
            return
        loaded = apply_workspace_paths(str(workspace_root), loaded)
    for _status, _progress, log in execute_config(stages, loaded):
        yield log


def make_component(action: argparse.Action, value: Any):
    label = f"--{action.dest}"
    info = action.help or "Pipeline setting."
    if action.dest in FIELD_GUIDANCE:
        info = f"{info} Impact: {FIELD_GUIDANCE[action.dest]}"
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
    initial, initial_stages, initial_workspace_root, initial_status = configuration_state()
    controls: list[Any] = []
    stage_tabs: list[Any] = []
    grouped = {name: set(keys) for name, keys in FIELD_GROUPS.items()}
    known_fields = {key for keys in grouped.values() for key in keys}

    with gr.Blocks(
        title="Anime2SD Frame Lab",
        elem_classes="workspace",
    ) as demo:
        with gr.Column(elem_classes="run-panel"):
            gr.HTML("<p class='panel-label'>Workflow</p>")
            stage_selector = gr.CheckboxGroup(
                choices=[(f"{number} - {details[0]}", str(number)) for number, details in STAGES.items()],
                value=initial_stages,
                label="Stages to run",
                info="Only enabled stages are run and shown in Settings below. Stage 3 uses character references.",
            )
            with gr.Accordion("Stage guide", open=False):
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
            run_status = gr.HTML(
                status_markup("Ready", "Select stages and run the pipeline.")
            )
            stage_progress = gr.HTML(
                progress_markup(stage_values(initial_stages), set(), None)
            )
            output = gr.Textbox(
                label="Run log", lines=18, interactive=False, elem_classes="run-log"
            )

        with gr.Accordion("Configuration", open=False, elem_classes="run-panel config-accordion"):
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="workspace-card"):
                        workspace_root = gr.Textbox(
                            label="Workspace root",
                            value=initial_workspace_root,
                            placeholder=r"C:\datasets\anime\my_project",
                            info=(
                                "Optional single working root. When set, runs and the global configuration use "
                                "<root>\\src, <root>\\dst, <root>\\ref, and <root>\\logs."
                            ),
                        )
                        create_workspace_button = gr.Button("Create workspace folders")
                        clear_output_button = gr.Button("Clear generated output", variant="stop")
                        gr.Markdown(
                            "`src` = first-stage input, `ref` = character reference images, "
                            "`dst/intermediate` and `dst/training` = generated pipeline data. "
                            "Creating a workspace keeps existing contents and leaves `dst` uncreated until "
                            "data is written; clearing output removes only `dst` results."
                        )
                with gr.Column():
                    uploaded = gr.File(
                        label="Import configuration",
                        file_types=[".toml"],
                        type="filepath",
                        elem_classes="compact-upload",
                    )
                    load_button = gr.Button("Load configuration")
                    gr.Markdown(
                        "One global TOML file stores stage selection, workspace paths, and all settings. "
                        "It is restored automatically when the application starts."
                    )
                    with gr.Row():
                        save_button = gr.Button("Save configuration", variant="primary")
                        export_button = gr.DownloadButton(
                            "Export settings",
                            value=str(GLOBAL_CONFIG) if GLOBAL_CONFIG.exists() else None,
                        )
            status = gr.Markdown(
                f"Web interface: `http://127.0.0.1:{PORT}` (fixed port).\n\n{initial_status}"
            )

        gr.HTML("<p class='settings-label'>Settings by stage</p>")
        with gr.Tabs(elem_classes="settings-tabs"):
            for group_name, fields in FIELD_GROUPS.items():
                stage_number = GROUP_STAGES.get(group_name)
                visible = stage_number is None or str(stage_number) in initial_stages
                with gr.Tab(group_name, visible=visible) as tab:
                    if stage_number is not None:
                        stage_tabs.append(tab)
                        gr.Markdown(STAGES[stage_number][1])
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

        shutdown_button = gr.Button("Shut down server", variant="stop", elem_classes="shutdown-button")

        ordered_controls = {action.dest: control for action, control in zip(
            [item for group in FIELD_GROUPS.values() for item in ACTIONS if item.dest in group] + remaining,
            controls,
        )}
        controls_in_action_order = [ordered_controls[action.dest] for action in ACTIONS]
        workspace_path_outputs = [
            ordered_controls["src_dir"],
            ordered_controls["dst_dir"],
            ordered_controls["character_ref_dir"],
            ordered_controls["log_dir"],
        ]

        create_workspace_button.click(
            create_workspace_structure,
            inputs=workspace_root,
            outputs=[*workspace_path_outputs, status],
            api_name="create_workspace",
        )
        clear_output_button.click(
            clear_workspace_output,
            inputs=workspace_root,
            outputs=status,
            api_name="clear_workspace_output",
        )

        save_button.click(
            save_configuration,
            inputs=[workspace_root, stage_selector, *controls_in_action_order],
            outputs=[status, export_button],
            api_name="save_configuration",
        )
        load_button.click(
            load_configuration,
            inputs=uploaded,
            outputs=[workspace_root, stage_selector, *controls_in_action_order, status],
        ).then(stage_tab_updates, inputs=stage_selector, outputs=stage_tabs)
        api_load_button = gr.Button(visible=False)
        api_load_button.click(
            load_configuration,
            outputs=[workspace_root, stage_selector, *controls_in_action_order, status],
            api_name="load_global_configuration",
        ).then(stage_tab_updates, inputs=stage_selector, outputs=stage_tabs)
        stage_selector.change(stage_tab_updates, inputs=stage_selector, outputs=stage_tabs)
        run_event = run_button.click(
            run_selected_stages,
            inputs=[stage_selector, workspace_root, *controls_in_action_order],
            outputs=[run_status, stage_progress, output],
            api_name="run_from_form",
            concurrency_limit=1,
            concurrency_id="pipeline_run",
        )
        stop_button.click(
            stop_pipeline,
            outputs=output,
            api_name="stop_pipeline",
            cancels=[run_event],
            concurrency_limit=None,
            concurrency_id="pipeline_control",
        )
        shutdown_button.click(
            shutdown_server,
            outputs=status,
            api_name="shutdown_server",
            concurrency_limit=None,
        )
        api_profile = gr.Textbox(value="configs/ui/configuration.toml", visible=False, container=False)
        api_stages = gr.Textbox(value="", visible=False, container=False)
        api_output = gr.Textbox(visible=False, container=False)
        api_button = gr.Button(visible=False)
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
