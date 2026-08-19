import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import anime2sd.classif.classify_characters as classification
import anime2sd.classif.file_utils as file_utils
import anime2sd.invalid_images as invalid_images
from anime2sd.common_preprocess import rearrange_related_files
from anime2sd.invalid_images import InvalidImageHandler, validate_image


def save_valid_image(path: Path, image_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color="white").save(path, format=image_format)


def save_truncated_image(path: Path, image_format: str) -> None:
    save_valid_image(path, image_format)
    content = path.read_bytes()
    path.write_bytes(content[: max(32, len(content) // 2)])
    with pytest.raises(invalid_images.EXPECTED_IMAGE_ERRORS):
        validate_image(path)


def make_handler(workspace: Path, **kwargs) -> InvalidImageHandler:
    return InvalidImageHandler(
        workspace,
        workspace / "logs",
        logger=logging.getLogger("invalid-image-test"),
        **kwargs,
    )


def decode_feature(path: str) -> np.ndarray:
    validate_image(path)
    return np.array([0.25, 0.5], dtype=np.float32)


def test_valid_jpg_is_processed_without_quarantine(tmp_path, monkeypatch):
    source = tmp_path / "src" / "valid.jpg"
    save_valid_image(source)
    handler = make_handler(tmp_path)
    monkeypatch.setattr(file_utils, "ccip_extract_feature", decode_feature)

    image_files, features, _characters, _mapping = (
        file_utils.load_image_features_and_characters(
            src_dir=str(source.parent),
            invalid_image_handler=handler,
            source_root=str(source.parent),
            save_ccip_cache=False,
        )
    )

    assert image_files.tolist() == [str(source)]
    assert features.shape == (1, 2)
    assert source.exists()
    assert handler.stage_stats(3).invalid == 0
    assert not (tmp_path / "quarantine").exists()


@pytest.mark.parametrize(
    ("filename", "image_format"),
    [("truncated.jpg", "JPEG"), ("truncated.png", "PNG")],
)
def test_truncated_images_are_quarantined_and_skipped(
    tmp_path, monkeypatch, filename, image_format
):
    source_root = tmp_path / "ref"
    source = source_root / "character_a" / filename
    save_truncated_image(source, image_format)
    handler = make_handler(tmp_path)
    monkeypatch.setattr(file_utils, "ccip_extract_feature", decode_feature)

    image_files, features, _characters, _mapping = (
        file_utils.load_image_features_and_characters(
            image_files=[str(source)],
            invalid_image_handler=handler,
            source_type="reference",
            source_root=str(source_root),
            validate_before_processing=True,
            save_ccip_cache=False,
        )
    )

    destination = tmp_path / "quarantine" / "stage_3" / "ref" / "character_a" / filename
    assert image_files.size == 0
    assert features.size == 0
    assert not source.exists()
    assert destination.exists()
    stats = handler.stage_stats(3)
    assert (stats.invalid, stats.moved, stats.move_failures) == (1, 1, 0)


def test_non_image_with_jpg_extension_is_quarantined(tmp_path, monkeypatch):
    source = tmp_path / "src" / "nested" / "fake.jpg"
    source.parent.mkdir(parents=True)
    source.write_text("not image data", encoding="utf-8")
    handler = make_handler(tmp_path)
    monkeypatch.setattr(file_utils, "ccip_extract_feature", decode_feature)

    result = file_utils.load_image_features_and_characters(
        src_dir=str(tmp_path / "src"),
        invalid_image_handler=handler,
        source_root=str(tmp_path / "src"),
        save_ccip_cache=False,
    )

    assert result[0].size == 0
    assert (
        tmp_path / "quarantine" / "stage_3" / "src" / "nested" / "fake.jpg"
    ).exists()


def test_preprocessing_quarantines_invalid_image_instead_of_aborting(tmp_path):
    source_root = tmp_path / "src"
    invalid = source_root / "episode" / "broken.jpg"
    invalid.parent.mkdir(parents=True)
    invalid.write_bytes(b"not an image")
    handler = make_handler(tmp_path)

    rearrange_related_files(
        str(source_root),
        invalid_image_handler=handler,
        stage=3,
    )

    assert not invalid.exists()
    assert (
        tmp_path / "quarantine" / "stage_3" / "src" / "episode" / "broken.jpg"
    ).exists()


def test_multiple_invalid_images_update_counters_and_jsonl(tmp_path, monkeypatch):
    source_root = tmp_path / "src"
    for index in range(3):
        path = source_root / f"bad_{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"invalid")
    handler = make_handler(tmp_path)
    monkeypatch.setattr(file_utils, "ccip_extract_feature", decode_feature)

    result = file_utils.load_image_features_and_characters(
        src_dir=str(source_root),
        invalid_image_handler=handler,
        source_root=str(source_root),
        save_ccip_cache=False,
    )

    stats = handler.stage_stats(3)
    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "invalid_images.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result[0].size == 0
    assert (stats.invalid, stats.moved, stats.move_failures) == (3, 3, 0)
    assert len(records) == 3
    assert all(record["stage"] == 3 and record["moved"] for record in records)
    assert "Operation: extract_dataset_feature" in (
        tmp_path / "logs" / "invalid_images.log"
    ).read_text(encoding="utf-8")


def test_quarantine_collision_does_not_overwrite_existing_file(tmp_path):
    source = tmp_path / "src" / "character" / "same.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new invalid file")
    existing = (
        tmp_path
        / "quarantine"
        / "stage_3"
        / "src"
        / "character"
        / "same.jpg"
    )
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing quarantine file")
    handler = make_handler(tmp_path)

    record = handler.handle_invalid_image(
        source,
        OSError("image file is truncated"),
        stage=3,
        operation="test_collision",
        source_type="dataset",
        source_root=tmp_path / "src",
    )

    assert existing.read_bytes() == b"existing quarantine file"
    assert Path(record.quarantine_path).name == "same_001.jpg"
    assert Path(record.quarantine_path).read_bytes() == b"new invalid file"


def test_concurrent_quarantine_moves_are_collision_safe(tmp_path):
    first = tmp_path / "input_a" / "same.jpg"
    second = tmp_path / "input_b" / "same.jpg"
    for path, content in ((first, b"first"), (second, b"second")):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    handler = make_handler(tmp_path)

    def quarantine(path: Path):
        return handler.handle_invalid_image(
            path,
            OSError("invalid"),
            stage=3,
            operation="concurrent_test",
            source_type="dataset",
            source_root=path.parent,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(quarantine, (first, second)))

    destinations = {Path(record.quarantine_path).name for record in records}
    assert destinations == {"same.jpg", "same_001.jpg"}
    assert handler.stage_stats(3).moved == 2
    assert len(
        (tmp_path / "logs" / "invalid_images.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2


def test_move_failure_is_logged_and_image_is_skipped(tmp_path, monkeypatch):
    source = tmp_path / "src" / "locked.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"invalid")
    handler = make_handler(tmp_path)
    monkeypatch.setattr(
        invalid_images.shutil,
        "move",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("locked")),
    )

    record = handler.handle_invalid_image(
        source,
        OSError("cannot decode"),
        stage=3,
        operation="extract_dataset_feature",
        source_type="dataset",
        source_root=tmp_path / "src",
    )

    assert source.exists()
    assert record.moved is False
    assert record.move_exception_type == "PermissionError"
    assert handler.stage_stats(3).move_failures == 1
    assert "Move exception: PermissionError" in (
        tmp_path / "logs" / "invalid_images.log"
    ).read_text(encoding="utf-8")


def test_valid_image_ccip_runtime_error_is_not_quarantined(tmp_path, monkeypatch):
    source = tmp_path / "src" / "valid.jpg"
    save_valid_image(source)
    handler = make_handler(tmp_path)
    monkeypatch.setattr(
        file_utils,
        "ccip_extract_feature",
        lambda _path: (_ for _ in ()).throw(OSError("ONNX Runtime failure")),
    )

    with pytest.raises(OSError, match="ONNX Runtime failure"):
        file_utils.load_image_features_and_characters(
            src_dir=str(source.parent),
            invalid_image_handler=handler,
            source_root=str(source.parent),
            save_ccip_cache=False,
        )

    assert source.exists()
    assert handler.stage_stats(3).invalid == 0


def test_bad_reference_between_valid_references_does_not_stop_stage_three(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "src"
    references = tmp_path / "ref" / "char1"
    for name in ("dataset_1.jpg", "dataset_2.jpg"):
        save_valid_image(source_root / name)
    save_valid_image(references / "001.jpg")
    (references / "002.jpg").write_bytes(b"invalid")
    save_valid_image(references / "003.jpg")
    handler = make_handler(tmp_path)
    monkeypatch.setattr(file_utils, "ccip_extract_feature", decode_feature)
    monkeypatch.setattr(
        classification,
        "_classify_feature_batch",
        lambda files, *_args: (np.zeros(len(files), dtype=int), {}),
    )
    monkeypatch.setattr(classification, "remove_empty_folders", lambda _path: None)
    saved = []

    def capture_save(files, _images, _dst, _labels, *_args, **_kwargs):
        saved.extend(Path(path).name for path in files)
        return {"char1": len(files)}

    monkeypatch.setattr(classification, "save_to_dir", capture_save)

    counts = classification.classify_from_directory(
        str(source_root),
        str(tmp_path / "dst"),
        ref_dir=str(tmp_path / "ref"),
        to_extract_from_noise=False,
        to_filter=False,
        clu_min_samples=2,
        invalid_image_handler=handler,
    )

    assert saved == ["dataset_1.jpg", "dataset_2.jpg"]
    assert counts == {"char1": 2}
    assert (references / "001.jpg").exists()
    assert not (references / "002.jpg").exists()
    assert (references / "003.jpg").exists()
    assert (
        tmp_path / "quarantine" / "stage_3" / "ref" / "char1" / "002.jpg"
    ).exists()


def test_stage_three_reports_when_no_valid_references_remain(tmp_path, monkeypatch):
    source = tmp_path / "src" / "dataset.jpg"
    reference = tmp_path / "ref" / "char1" / "bad.jpg"
    save_valid_image(source)
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"invalid")
    handler = make_handler(tmp_path)
    monkeypatch.setattr(file_utils, "ccip_extract_feature", decode_feature)

    with pytest.raises(RuntimeError, match="0 valid reference images remain"):
        classification.classify_from_directory(
            str(source.parent),
            str(tmp_path / "dst"),
            ref_dir=str(tmp_path / "ref"),
            invalid_image_handler=handler,
        )

    assert handler.invalid_count(3, "reference") == 1
