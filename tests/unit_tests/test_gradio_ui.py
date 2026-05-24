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
    assert "Save settings to profile" in components
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


def test_stop_pipeline_terminates_active_child_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with ui.PIPELINE_PROCESS_LOCK:
            ui.ACTIVE_PIPELINE_PROCESS = process

        assert ui.stop_pipeline() == "Stopping the active pipeline..."
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
