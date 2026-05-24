import os
import subprocess
import sys

import app.gradio_ui as ui


def test_interface_exposes_control_endpoints_and_stage_tabs():
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
    assert "Workspace root" in components
    assert "Create workspace folders" in components
    assert "Save profile" in components
    assert "Save settings to profile" not in components
    assert "create_workspace" in dependencies
    assert "stop_pipeline" in dependencies
    assert "shutdown_server" in dependencies
    assert tabs[0] == "General"
    assert tabs[-1] == "Stage 7 - Balance"
    tab_props = {
        component.get("props", {}).get("label"): component.get("props", {})
        for component in config["components"]
        if component.get("type") == "tabitem"
    }
    assert tab_props["Stage 0 - Download"]["visible"] is False
    assert tab_props["Stage 3 - Classify"]["visible"] is True


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
    assert "Workspace ready" in status
    for relative_path in ui.WORKSPACE_DIRECTORIES:
        assert (tmp_path / relative_path).is_dir()


def test_consecutive_selected_stages_are_run_as_one_pipeline_segment():
    assert ui.stage_ranges([2, 3, 5, 6, 7]) == [(2, 3), (5, 7)]


def test_saved_profile_records_the_starting_preset_with_ui_settings(tmp_path, monkeypatch):
    values = [
        ui.component_value(action, ui.defaults().get(action.dest))
        for action in ui.ACTIONS
    ]
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "SAVED_CONFIG_DIR", tmp_path / "configs" / "ui" / "saved")

    _status, output_path = ui.save_configuration(
        "combined",
        "configs/pipelines/booru.toml",
        r"C:\datasets\anime\project",
        ["2", "3"],
        *values,
    )
    saved = ui.toml.load(output_path)

    assert saved["ui"]["source_preset"] == "configs/pipelines/booru.toml"
    assert saved["ui"]["enabled_stages"] == [2, 3]
    assert saved["ui"]["workspace_root"] == os.path.normpath(r"C:\datasets\anime\project")


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
