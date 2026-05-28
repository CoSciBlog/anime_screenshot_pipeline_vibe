import os
import subprocess
import sys

import app.gradio_ui as ui


def test_interface_exposes_control_endpoints_and_stage_tabs(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "GLOBAL_CONFIG", tmp_path / "missing.toml")
    config = ui.build_interface().get_config_file()
    components = " ".join(str(component.get("props", {})) for component in config["components"])
    dependencies = " ".join(str(dependency) for dependency in config["dependencies"])
    tabs = [
        component.get("props", {}).get("label")
        for component in config["components"]
        if component.get("type") == "tabitem"
    ]

    assert "Stop pipeline" in components
    assert "Shut down server" in components
    assert "Yes, shut down server" in components
    assert "Shut down the server?" in components
    assert "Workspace root" in components
    assert "Create workspace folders" in components
    assert "Clear generated output" in components
    assert "Save configuration" in components
    assert "Export settings" in components
    assert "Starting preset" not in components
    assert "Configuration name" not in components
    assert "Import configuration" in components
    assert "2 - Detect" in components
    assert "2 - Crop" not in components
    assert "Stage guide" in components
    assert "Configuration" in components
    assert "Programmatic access" not in components
    assert "--max_images_per_character" in components
    assert "--max_images_per_character_per_episode" in components
    assert "--remove_classified_aux_files" in components
    assert "--remove_stage2_crops_after_classification" in components
    assert "--remove_noise_folder_after_classification" in components
    assert "--classification_chunk_size" in components
    assert "create_workspace" in dependencies
    assert "clear_workspace_output" in dependencies
    assert "run_saved_profile" in dependencies
    assert "stop_pipeline" in dependencies
    assert "shutdown_server" in dependencies
    assert "load_global_configuration" in dependencies
    assert tabs[0] == "General"
    assert tabs[-1] == "Stage 7 - Balance"
    tab_props = {
        component.get("props", {}).get("label"): component.get("props", {})
        for component in config["components"]
        if component.get("type") == "tabitem"
    }
    assert tab_props["Stage 0 - Download"]["visible"] is False
    assert tab_props["Stage 2 - Detect"]["visible"] is False
    assert tab_props["Stage 3 - Classify"]["visible"] is True
    shutdown_index = next(
        index for index, component in enumerate(config["components"])
        if component.get("type") == "button"
        and component.get("props", {}).get("value") == "Shut down server"
    )
    save_indices = [
        index for index, component in enumerate(config["components"])
        if component.get("type") == "button"
        and component.get("props", {}).get("value") == "Save configuration"
    ]
    confirm_shutdown_id = next(
        component["id"] for component in config["components"]
        if component.get("type") == "button"
        and component.get("props", {}).get("value") == "Yes, shut down server"
    )
    shutdown_dependency = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("api_name") == "shutdown_server"
    )
    last_stage_tab_index = max(
        index for index, component in enumerate(config["components"])
        if component.get("type") == "tabitem"
    )
    assert len(save_indices) == 2
    assert max(save_indices) > last_stage_tab_index
    assert shutdown_index > last_stage_tab_index
    assert shutdown_dependency["targets"][0][0] == confirm_shutdown_id


def test_stage_settings_visibility_follows_enabled_stages():
    updates = ui.stage_tab_updates(["0", "3"])

    assert updates[0]["visible"] is True
    assert updates[1]["visible"] is False
    assert updates[3]["visible"] is True


def test_path_configuration_values_are_normalized_for_host_platform():
    ref_action = next(action for action in ui.ACTIONS if action.dest == "character_ref_dir")
    entered = r"C:/datasets/anime/references/frieren"

    assert ui.config_value(ref_action, entered) == os.path.normpath(entered)


def test_workspace_structure_maps_and_creates_all_pipeline_directories(tmp_path):
    src, dst, ref, logs, status = ui.create_workspace_structure(str(tmp_path))

    assert src["value"] == str(tmp_path / "src")
    assert dst["value"] == str(tmp_path / "dst")
    assert ref["value"] == str(tmp_path / "ref")
    assert logs["value"] == str(tmp_path / "logs")
    assert "Workspace created" in status
    for relative_path in ui.WORKSPACE_DIRECTORIES:
        assert (tmp_path / relative_path).is_dir()
    assert not (tmp_path / "dst").exists()


def test_workspace_structure_preserves_existing_contents(tmp_path):
    existing_file = tmp_path / "src" / "existing.png"
    existing_file.parent.mkdir()
    existing_file.write_text("keep", encoding="utf-8")

    *_updates, status = ui.create_workspace_structure(str(tmp_path))

    assert existing_file.read_text(encoding="utf-8") == "keep"
    assert "Existing folders and their contents were kept" in status


def test_workspace_structure_rejects_dst_file_without_creating_output_tree(tmp_path):
    (tmp_path / "dst").write_text("not a directory", encoding="utf-8")

    *_updates, status = ui.create_workspace_structure(str(tmp_path))

    assert "exists but is not a folder" in status
    assert not (tmp_path / "src").exists()


def test_clear_workspace_output_deletes_only_generated_dst_content(tmp_path):
    source = tmp_path / "src" / "keep.png"
    reference = tmp_path / "ref" / "frieren" / "keep.png"
    result = tmp_path / "dst" / "training" / "delete.webp"
    for file_path in (source, reference, result):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("data", encoding="utf-8")

    status = ui.clear_workspace_output(str(tmp_path))

    assert source.exists()
    assert reference.exists()
    assert not result.exists()
    assert not (tmp_path / "dst").exists()
    assert "Generated output cleared" in status


def test_clear_workspace_output_refuses_while_pipeline_is_running(tmp_path):
    result = tmp_path / "dst" / "training" / "keep.webp"
    result.parent.mkdir(parents=True)
    result.write_text("data", encoding="utf-8")

    class RunningProcess:
        def poll(self):
            return None

    with ui.PIPELINE_PROCESS_LOCK:
        ui.ACTIVE_PIPELINE_PROCESS = RunningProcess()
    try:
        status = ui.clear_workspace_output(str(tmp_path))
    finally:
        with ui.PIPELINE_PROCESS_LOCK:
            ui.ACTIVE_PIPELINE_PROCESS = None

    assert result.exists()
    assert "Stop the active pipeline" in status


def test_run_output_is_mirrored_to_terminal(capsys):
    ui.mirror_run_output("running stage\n")

    assert capsys.readouterr().out == "running stage\n"


def test_run_log_cleaning_replaces_repeated_progress_lines():
    history = []

    ui.append_run_history(history, "Extract dataset features: 40%|#### | 4/10 [00:01<00:02]")
    ui.append_run_history(history, "Extract dataset features: 50%|#####| 5/10 [00:02<00:02]")

    assert ui.clean_run_line("old progress\r\x1b[Avisible\r\n") == "visible"
    assert history == ["Extract dataset features: 50%|#####| 5/10 [00:02<00:02]"]
    assert "50%" in ui.run_detail_from_line(history[0])


def test_stage_progress_reports_completed_and_current_stage():
    markup = ui.progress_markup([2, 3, 4], {2}, 3, "Clustering active")

    assert "Current: Stage 3 - Classify" in markup
    assert "1 of 3 stages complete" in markup
    assert "Clustering active" in markup


def test_consecutive_selected_stages_are_run_as_one_pipeline_segment():
    assert ui.stage_ranges([2, 3, 5, 6, 7]) == [(2, 3), (5, 7)]


def test_global_configuration_records_stages_and_exports_same_file(tmp_path, monkeypatch):
    values = [
        ui.component_value(action, ui.defaults().get(action.dest))
        for action in ui.ACTIONS
    ]
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "GLOBAL_CONFIG", tmp_path / "configs" / "ui" / "configuration.toml")

    _status, export_path = ui.save_configuration(
        r"C:\datasets\anime\project",
        ["2", "3"],
        *values,
    )
    output_path = ui.GLOBAL_CONFIG
    saved = ui.toml.load(output_path)

    assert saved["ui"]["enabled_stages"] == [2, 3]
    assert saved["ui"]["workspace_root"] == os.path.normpath(r"C:\datasets\anime\project")
    assert export_path == str(output_path)


def test_global_configuration_records_stage_three_cleanup_toggles(tmp_path, monkeypatch):
    values = [
        ui.component_value(action, ui.defaults().get(action.dest))
        for action in ui.ACTIONS
    ]
    stage2_cleanup_index = next(
        index for index, action in enumerate(ui.ACTIONS)
        if action.dest == "remove_stage2_crops_after_classification"
    )
    noise_cleanup_index = next(
        index for index, action in enumerate(ui.ACTIONS)
        if action.dest == "remove_noise_folder_after_classification"
    )
    values[stage2_cleanup_index] = True
    values[noise_cleanup_index] = True
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "GLOBAL_CONFIG", tmp_path / "configs" / "ui" / "configuration.toml")

    ui.save_configuration("", ["3"], *values)
    saved = ui.toml.load(ui.GLOBAL_CONFIG)

    assert saved["remove_stage2_crops_after_classification"] is True
    assert saved["remove_noise_folder_after_classification"] is True


def test_global_configuration_is_loaded_with_stages_and_settings_at_startup(tmp_path, monkeypatch):
    path = tmp_path / "configs" / "ui" / "configuration.toml"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        ui.toml.dump(
            {
                "tag_threshold": "0.75",
                "classification_chunk_size": "1024",
                "ui": {
                    "enabled_stages": [2, 5],
                    "workspace_root": r"C:\datasets\loaded",
                },
            },
            handle,
        )
    monkeypatch.setattr(ui, "GLOBAL_CONFIG", path)

    updates = ui.load_configuration(None)
    tag_threshold_index = next(
        index for index, action in enumerate(ui.ACTIONS) if action.dest == "tag_threshold"
    )
    classification_chunk_size_index = next(
        index for index, action in enumerate(ui.ACTIONS) if action.dest == "classification_chunk_size"
    )
    min_download_episode_index = next(
        index for index, action in enumerate(ui.ACTIONS) if action.dest == "min_download_episode"
    )

    assert updates[0]["value"] == r"C:\datasets\loaded"
    assert updates[1]["value"] == ["2", "5"]
    assert updates[tag_threshold_index + 2]["value"] == 0.75
    assert updates[classification_chunk_size_index + 2]["value"] == 1024
    assert updates[min_download_episode_index + 2]["value"] is None
    assert "Global configuration loaded" in updates[-1]


def test_initial_interface_uses_global_configuration_without_page_load_callback(tmp_path, monkeypatch):
    path = tmp_path / "configs" / "ui" / "configuration.toml"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        ui.toml.dump(
            {
                "tag_threshold": "0.81",
                "ui": {
                    "enabled_stages": [2, 5],
                    "workspace_root": r"C:\datasets\loaded",
                },
            },
            handle,
        )
    monkeypatch.setattr(ui, "GLOBAL_CONFIG", path)

    config = ui.build_interface().get_config_file()
    props = [
        component.get("props", {})
        for component in config["components"]
    ]
    stage_selector = next(item for item in props if item.get("label") == "Stages to run")
    workspace_root = next(item for item in props if item.get("label") == "Workspace root")
    tag_threshold = next(item for item in props if item.get("label") == "--tag_threshold")
    stage_two = next(item for item in props if item.get("label") == "Stage 2 - Detect")
    stage_three = next(item for item in props if item.get("label") == "Stage 3 - Classify")
    load_global = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("api_name") == "load_global_configuration"
    )

    assert stage_selector["value"] == ["2", "5"]
    assert workspace_root["value"] == r"C:\datasets\loaded"
    assert tag_threshold["value"] == 0.81
    assert stage_two["visible"] is True
    assert stage_three["visible"] is False
    assert all(target[1] != "load" for target in load_global["targets"])
    assert not any(
        target[1] == "load"
        for dependency in config["dependencies"]
        for target in dependency.get("targets", [])
    )


def test_running_from_form_creates_missing_workspace_folders(tmp_path, monkeypatch):
    captured = []

    def fake_execute(stages, _config):
        captured.extend(stages)
        yield "status", "progress", "log"

    monkeypatch.setattr(ui, "execute_config", fake_execute)
    values = [
        ui.component_value(action, ui.defaults().get(action.dest))
        for action in ui.ACTIONS
    ]

    result = list(ui.run_selected_stages(["2", "3"], str(tmp_path), *values))

    assert result == [("status", "progress", "log")]
    assert captured == [2, 3]
    assert (tmp_path / "logs").is_dir()
    assert not (tmp_path / "dst").exists()


def test_stop_pipeline_terminates_active_child_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with ui.PIPELINE_PROCESS_LOCK:
            ui.ACTIVE_PIPELINE_PROCESS = process

        assert ui.stop_pipeline() == "Pipeline stopped."
        assert process.poll() is not None
        assert ui.stop_pipeline() == "No pipeline process is currently running."
    finally:
        if process.poll() is None:
            process.kill()
        with ui.PIPELINE_PROCESS_LOCK:
            ui.ACTIVE_PIPELINE_PROCESS = None
        ui.PIPELINE_STOP_REQUESTED.clear()


def test_shutdown_server_requests_pipeline_stop_and_process_exit(monkeypatch):
    calls = []

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    monkeypatch.setattr(ui, "stop_pipeline", lambda: calls.append("stop"))
    monkeypatch.setattr(ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(ui.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(ui.os, "_exit", lambda code: calls.append(code))

    assert ui.shutdown_server().startswith("Server shutdown requested.")
    assert calls == ["stop", 0]


def test_shutdown_confirmation_visibility_updates():
    assert ui.show_shutdown_confirmation()["visible"] is True
    assert ui.hide_shutdown_confirmation()["visible"] is False
