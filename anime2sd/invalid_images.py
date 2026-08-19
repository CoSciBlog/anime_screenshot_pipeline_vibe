"""Central handling for invalid image inputs.

The handler deliberately separates image decoding failures from model/runtime
failures.  Callers should only quarantine a file after Pillow has confirmed
that the image itself cannot be decoded.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


EXPECTED_IMAGE_ERRORS = (
    UnidentifiedImageError,
    Image.DecompressionBombError,
    OSError,
    EOFError,
)

STAGE_NAMES = {
    0: "Download",
    1: "Frame Extraction",
    2: "Character Detection",
    3: "Character Classification",
    4: "Dataset Selection",
    5: "Tagging and Captioning",
    6: "Arrangement",
    7: "Balancing",
}

SOURCE_DIRECTORY_NAMES = {
    "reference": "ref",
    "ref": "ref",
    "dataset": "src",
    "source": "src",
    "src": "src",
    "intermediate": "intermediate",
}

_QUARANTINE_IO_LOCK = threading.RLock()


@dataclass(frozen=True)
class InvalidImageRecord:
    timestamp: str
    stage: int
    operation: str
    source_type: str
    original_path: str
    quarantine_path: Optional[str]
    exception_type: str
    exception_message: str
    moved: bool
    move_exception_type: Optional[str] = None
    move_exception_message: Optional[str] = None


@dataclass
class StageImageStats:
    attempted: int = 0
    successful: int = 0
    invalid: int = 0
    moved: int = 0
    move_failures: int = 0
    invalid_by_source: Counter = field(default_factory=Counter)


def validate_image(path: str | Path) -> None:
    """Fully decode an image without accepting truncated data."""
    with Image.open(path) as image:
        image.load()


def load_decoded_image(path: str | Path) -> Image.Image:
    """Fully decode an image and detach it from its file handle."""
    with Image.open(path) as image:
        image.load()
        return image.copy()


def infer_workspace_root(config) -> Path:
    """Infer a stable workspace root from existing pipeline configuration."""
    explicit = getattr(config, "workspace_root", None)
    if explicit:
        return Path(explicit).expanduser().resolve()

    preferred = (
        (getattr(config, "log_dir", None), "logs"),
        (getattr(config, "character_ref_dir", None), "ref"),
        (getattr(config, "src_dir", None), "src"),
        (getattr(config, "dst_dir", None), "dst"),
    )
    for value, expected_name in preferred:
        if value and str(value).lower() != "none":
            path = Path(value).expanduser().resolve()
            if path.name.lower() == expected_name:
                return path.parent

    candidates = [
        Path(value).expanduser().resolve()
        for value in (
            getattr(config, "src_dir", None),
            getattr(config, "dst_dir", None),
            getattr(config, "character_ref_dir", None),
            getattr(config, "log_dir", None),
        )
        if value and str(value).lower() != "none"
    ]
    if not candidates:
        return Path.cwd().resolve()
    common_parts = list(candidates[0].parts)
    for candidate in candidates[1:]:
        prefix = []
        for left, right in zip(common_parts, candidate.parts):
            if left.casefold() != right.casefold():
                break
            prefix.append(left)
        common_parts = prefix
    if common_parts:
        common = Path(*common_parts)
        if common != Path(common.anchor):
            return common
    return candidates[0].parent


class InvalidImageHandler:
    """Thread-safe quarantine, logging, and statistics for invalid images."""

    def __init__(
        self,
        workspace_root: str | Path,
        log_dir: str | Path | None,
        *,
        quarantine_invalid_images: bool = True,
        continue_on_invalid_image: bool = True,
        quarantine_dir: str | Path = "auto",
        invalid_image_log: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.quarantine_invalid_images = quarantine_invalid_images
        self.continue_on_invalid_image = continue_on_invalid_image
        self.invalid_image_log = invalid_image_log
        self.logger = logger or logging.getLogger(__name__)
        self.quarantine_root = (
            self.workspace_root / "quarantine"
            if not quarantine_dir or str(quarantine_dir).strip().lower() == "auto"
            else Path(quarantine_dir).expanduser().resolve()
        )
        self.log_dir = (
            None
            if not log_dir or str(log_dir).strip().lower() == "none"
            else Path(log_dir).expanduser().resolve()
        )
        self._lock = threading.RLock()
        self._stats: Dict[int, StageImageStats] = defaultdict(StageImageStats)
        self._records: list[InvalidImageRecord] = []
        self._stage_durations: Dict[int, float] = {}

    @classmethod
    def from_config(cls, config, logger: Optional[logging.Logger] = None):
        return cls(
            infer_workspace_root(config),
            getattr(config, "log_dir", None),
            quarantine_invalid_images=getattr(
                config, "quarantine_invalid_images", True
            ),
            continue_on_invalid_image=getattr(
                config, "continue_on_invalid_image", True
            ),
            quarantine_dir=getattr(config, "quarantine_dir", "auto"),
            invalid_image_log=getattr(config, "invalid_image_log", True),
            logger=logger,
        )

    def record_attempt(self, stage: int) -> None:
        with self._lock:
            self._stats[stage].attempted += 1

    def record_success(self, stage: int) -> None:
        with self._lock:
            self._stats[stage].successful += 1

    def stage_stats(self, stage: int) -> StageImageStats:
        with self._lock:
            stats = self._stats[stage]
            return StageImageStats(
                attempted=stats.attempted,
                successful=stats.successful,
                invalid=stats.invalid,
                moved=stats.moved,
                move_failures=stats.move_failures,
                invalid_by_source=Counter(stats.invalid_by_source),
            )

    def invalid_count(self, stage: int, source_type: Optional[str] = None) -> int:
        stats = self.stage_stats(stage)
        return (
            stats.invalid_by_source[source_type]
            if source_type is not None
            else stats.invalid
        )

    def _source_directory(self, source_type: str) -> str:
        normalized = source_type.strip().lower().replace(" ", "_")
        return SOURCE_DIRECTORY_NAMES.get(normalized, normalized or "other")

    def _relative_path(self, original: Path, source_root: Optional[str | Path]) -> Path:
        root = Path(source_root).expanduser().resolve() if source_root else None
        if root is not None:
            try:
                return original.resolve().relative_to(root)
            except ValueError:
                pass
        return Path(original.name)

    def _available_destination(self, requested: Path) -> Path:
        if not requested.exists():
            return requested
        for index in range(1, 1_000_000):
            candidate = requested.with_name(
                f"{requested.stem}_{index:03d}{requested.suffix}"
            )
            if not candidate.exists():
                return candidate
        raise OSError(f"No available quarantine filename for {requested}")

    def _append_logs(self, record: InvalidImageRecord) -> None:
        if not self.invalid_image_log or self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.log_dir / "invalid_images.log"
        jsonl_path = self.log_dir / "invalid_images.jsonl"
        text = (
            f"{record.timestamp.replace('T', ' ')}\n"
            f"Stage: {record.stage}\n"
            f"Operation: {record.operation}\n"
            f"Source: {record.source_type}\n"
            f"Original: {record.original_path}\n"
            f"Quarantine: {record.quarantine_path or '-'}\n"
            f"Exception: {record.exception_type}\n"
            f"Message: {record.exception_message}\n"
            f"Moved: {'yes' if record.moved else 'no'}\n"
        )
        if record.move_exception_type:
            text += (
                f"Move exception: {record.move_exception_type}\n"
                f"Move message: {record.move_exception_message}\n"
            )
        with text_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _log_normal_file(self, message: str) -> None:
        """Write to configured normal file handlers without duplicating terminal text."""
        record = self.logger.makeRecord(
            self.logger.name,
            logging.WARNING,
            __file__,
            0,
            message,
            (),
            None,
        )
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.handle(record)

    def handle_invalid_image(
        self,
        path: str | Path,
        exc: BaseException,
        *,
        stage: int,
        operation: str,
        source_type: str,
        source_root: Optional[str | Path] = None,
    ) -> InvalidImageRecord:
        """Record and optionally quarantine one confirmed decode failure."""
        original = Path(path).expanduser().resolve()
        destination: Optional[Path] = None
        moved = False
        move_error: Optional[BaseException] = None
        log_error: Optional[BaseException] = None
        with _QUARANTINE_IO_LOCK, self._lock:
            if self.quarantine_invalid_images:
                requested = (
                    self.quarantine_root
                    / f"stage_{stage}"
                    / self._source_directory(source_type)
                    / self._relative_path(original, source_root)
                )
                try:
                    requested.parent.mkdir(parents=True, exist_ok=True)
                    destination = self._available_destination(requested)
                    shutil.move(str(original), str(destination))
                    moved = True
                except (OSError, shutil.Error) as error:
                    move_error = error
                    destination = destination or requested

            record = InvalidImageRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                stage=stage,
                operation=operation,
                source_type=source_type,
                original_path=str(original),
                quarantine_path=str(destination) if destination else None,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                moved=moved,
                move_exception_type=type(move_error).__name__ if move_error else None,
                move_exception_message=str(move_error) if move_error else None,
            )
            stats = self._stats[stage]
            stats.invalid += 1
            stats.invalid_by_source[source_type] += 1
            if moved:
                stats.moved += 1
            elif self.quarantine_invalid_images:
                stats.move_failures += 1
            self._records.append(record)
            try:
                self._append_logs(record)
            except OSError as error:
                log_error = error

        action = "moved to quarantine" if moved else "skipped"
        lines = [
            "[WARNING] Invalid image skipped",
            f"Stage:       {stage} - {STAGE_NAMES.get(stage, 'Pipeline Stage')}",
            f"Source:      {source_type}",
            f"File:        {original}",
            f"Error:       {type(exc).__name__}: {exc}",
            f"Action:      {action}",
        ]
        if destination:
            lines.append(f"Destination: {destination}")
        if move_error:
            lines.extend(
                [
                    "[WARNING] Invalid image could not be moved",
                    f"Move error:  {type(move_error).__name__}: {move_error}",
                ]
            )
        if log_error:
            lines.extend(
                [
                    "[WARNING] Invalid-image log could not be written",
                    f"Log error:   {type(log_error).__name__}: {log_error}",
                ]
            )
        message = "\n".join(lines)
        tqdm.write(message)
        self._log_normal_file(message)
        if not self.continue_on_invalid_image:
            raise exc
        return record

    def try_validate(
        self,
        path: str | Path,
        *,
        stage: int,
        operation: str,
        source_type: str,
        source_root: Optional[str | Path] = None,
    ) -> bool:
        self.record_attempt(stage)
        try:
            validate_image(path)
        except EXPECTED_IMAGE_ERRORS as exc:
            self.handle_invalid_image(
                path,
                exc,
                stage=stage,
                operation=operation,
                source_type=source_type,
                source_root=source_root,
            )
            return False
        self.record_success(stage)
        return True

    def set_stage_duration(self, stage: int, seconds: float) -> None:
        with self._lock:
            self._stage_durations[stage] = seconds

    def log_stage_summary(
        self, stage: int, seconds: float, *, success: bool = True
    ) -> None:
        self.set_stage_duration(stage, seconds)
        stats = self.stage_stats(stage)
        name = STAGE_NAMES.get(stage, "Pipeline Stage")
        lines = [
            "=" * 60,
            f"[SUMMARY] Stage {stage} - {name} {'finished' if success else 'failed'}",
            "=" * 60,
            f"Duration:                   {seconds:.2f}s",
            f"Images processed:          {stats.attempted}",
            f"Successfully processed:    {stats.successful}",
            f"Skipped invalid images:    {stats.invalid}",
            f"Moved to quarantine:       {stats.moved}",
            f"Failed quarantine moves:   {stats.move_failures}",
        ]
        if stage == 3:
            lines.extend(
                [
                    f"Invalid reference images: {stats.invalid_by_source['reference']}",
                    f"Invalid dataset images:   {stats.invalid_by_source['dataset']}",
                ]
            )
        if stats.invalid:
            lines.extend(["", "Quarantine:", str(self.quarantine_root / f"stage_{stage}")])
        elif success:
            lines.extend(["No invalid images detected."])
        lines.append("Pipeline continues normally." if success else "Pipeline stopped for a critical error.")
        lines.append("=" * 60)
        self.logger.info("\n".join(lines))

    def log_pipeline_summary(
        self,
        stages: Iterable[int],
        total_seconds: float,
        character_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        lines = ["=" * 60, "[SUMMARY] PIPELINE COMPLETE", "=" * 60]
        total_invalid = total_moved = total_move_failures = 0
        for stage in stages:
            stats = self.stage_stats(stage)
            duration = self._stage_durations.get(stage, 0.0)
            lines.append(
                f"Stage {stage}: {duration:.2f}s | invalid {stats.invalid} | "
                f"quarantined {stats.moved}"
            )
            if stage == 3 and stats.invalid:
                lines.append(
                    "  Reference images: "
                    f"{stats.invalid_by_source['reference']} | Dataset images: "
                    f"{stats.invalid_by_source['dataset']}"
                )
            total_invalid += stats.invalid
            total_moved += stats.moved
            total_move_failures += stats.move_failures
        lines.extend(
            [
                "-" * 60,
                f"Pipeline duration:          {total_seconds:.2f}s",
                f"Total invalid images:       {total_invalid}",
                f"Moved to quarantine:        {total_moved}",
                f"Move failures:              {total_move_failures}",
            ]
        )
        if character_counts:
            lines.extend(["-" * 60, "Recognized images per character:"])
            lines.extend(
                f"  {name}: {count}"
                for name, count in sorted(character_counts.items(), key=lambda item: item[0].casefold())
            )
        lines.extend(
            [
                "-" * 60,
                "Quarantine directory:",
                str(self.quarantine_root),
                "[OK] Pipeline completed successfully.",
                "=" * 60,
            ]
        )
        self.logger.info("\n".join(lines))
