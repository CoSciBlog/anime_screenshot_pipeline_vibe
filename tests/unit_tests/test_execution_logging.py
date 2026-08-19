import io
import logging

from anime2sd.execution_ordering import ColorFormatter, supports_terminal_color


class TerminalStream(io.StringIO):
    def isatty(self):
        return True


def test_color_formatter_adds_severity_color_when_enabled():
    formatter = ColorFormatter("%(levelname)s - %(message)s", use_color=True)
    record = logging.LogRecord("test", logging.ERROR, "", 0, "failed", (), None)

    assert formatter.format(record) == "\033[31mERROR - failed\033[0m"


def test_color_formatter_keeps_pipe_output_plain():
    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord("test", logging.INFO, "", 0, "running", (), None)

    assert formatter.format(record) == "INFO - running"


def test_no_color_environment_disables_terminal_colors(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert supports_terminal_color(TerminalStream()) is False


def test_interactive_terminal_enables_colors(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert supports_terminal_color(TerminalStream()) is True


def test_summary_and_ok_messages_have_distinct_colors():
    formatter = ColorFormatter("%(levelname)s - %(message)s", use_color=True)
    summary = logging.LogRecord("test", logging.INFO, "", 0, "[SUMMARY] done", (), None)
    success = logging.LogRecord("test", logging.INFO, "", 0, "[OK] done", (), None)

    assert formatter.format(summary) == "\033[1;36mINFO - [SUMMARY] done\033[0m"
    assert formatter.format(success) == "\033[32mINFO - [OK] done\033[0m"
