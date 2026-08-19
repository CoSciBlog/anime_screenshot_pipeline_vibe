import json
import importlib
import logging
import os

import numpy as np
from PIL import Image
from waifuc.source.base import BaseDataSource
from waifuc.model import ImageItem

import anime2sd.stage2_cropping as cropping
from anime2sd.basics import get_corr_meta_names
from anime2sd.classif import file_utils
from anime2sd.classif.classify_characters import classify_from_directory
from anime2sd.image_selection import resize_character_images
from anime2sd.parse_arguments import create_parser
from anime2sd.stage2_cropping import (
    SingleCharacterAwarePersonSplitAction,
    Stage2CropStats,
    log_stage2_crop_summary,
)
from anime2sd.waifuc_customize import SaveExporter


classification = importlib.import_module("anime2sd.classif.classify_characters")
merge_clusters_module = importlib.import_module("anime2sd.classif.merge_clusters")


def make_item(size=(100, 100)):
    return ImageItem(
        Image.new("RGB", size, "white"),
        {
            "filename": "frame.png",
            "current_path": os.path.abspath("frame.png"),
            "path": os.path.abspath("frame.png"),
            "image_size": list(size),
            "tags": ["solo"],
        },
    )


def detection(area=(10, 10, 60, 90), score=0.9):
    return area, "person", score


def action(**kwargs):
    return SingleCharacterAwarePersonSplitAction(
        min_crop_size=0,
        person_conf={"level": "n"},
        **kwargs,
    )


def test_parser_defaults_keep_legacy_stage2_behavior():
    args = create_parser().parse_args([])

    assert args.keep_single_character_uncropped is False
    assert args.single_character_uncropped_min_area_ratio == 0.0


def test_option_disabled_produces_normal_character_crop(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cropping,
        "detect_person",
        lambda *args, **kwargs: calls.append(1) or [detection()],
    )

    outputs = list(action(keep_single_character_uncropped=False).iter(make_item()))

    assert len(outputs) == 1
    assert outputs[0].image.size == (50, 80)
    assert outputs[0].meta["filename"] == "frame_person0.png"
    assert outputs[0].meta["character_crop_mode"] == "person_crop"
    assert len(calls) == 1


def test_one_person_keeps_exactly_one_full_original(monkeypatch):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])

    outputs = list(action(keep_single_character_uncropped=True).iter(make_item()))

    assert len(outputs) == 1
    assert outputs[0].image.size == (100, 100)
    assert outputs[0].meta["filename"] == "frame.png"
    assert outputs[0].meta["character_crop_mode"] == "uncropped_single"
    assert outputs[0].meta["person_count"] == 1
    assert outputs[0].meta["person_bbox"] == [10.0, 10.0, 60.0, 90.0]
    assert outputs[0].meta["tags"] == ["solo"]


def test_two_people_produce_two_crops(monkeypatch):
    monkeypatch.setattr(
        cropping,
        "detect_person",
        lambda *args, **kwargs: [
            detection((0, 0, 40, 100)),
            detection((50, 0, 100, 100)),
        ],
    )

    outputs = list(action(keep_single_character_uncropped=True).iter(make_item()))

    assert [output.image.size for output in outputs] == [(40, 100), (50, 100)]
    assert [output.meta["filename"] for output in outputs] == [
        "frame_person0.png",
        "frame_person1.png",
    ]
    assert all(output.meta["character_crop_mode"] == "person_crop" for output in outputs)


def test_three_people_produce_three_crops(monkeypatch):
    monkeypatch.setattr(
        cropping,
        "detect_person",
        lambda *args, **kwargs: [
            detection((0, 0, 30, 100)),
            detection((35, 0, 65, 100)),
            detection((70, 0, 100, 100)),
        ],
    )

    outputs = list(action(keep_single_character_uncropped=True).iter(make_item()))

    assert len(outputs) == 3
    assert [output.meta["person_count"] for output in outputs] == [3, 3, 3]


def test_no_person_keeps_existing_no_output_behavior(monkeypatch):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [])

    crop_action = action(keep_single_character_uncropped=True)

    assert list(crop_action.iter(make_item())) == []
    assert crop_action.stats.snapshot()["images_without_valid_character"] == 1


def test_min_area_ratio_keeps_original_when_threshold_is_met(monkeypatch):
    monkeypatch.setattr(
        cropping,
        "detect_person",
        lambda *args, **kwargs: [detection((0, 0, 50, 60))],
    )

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            single_character_uncropped_min_area_ratio=0.10,
        ).iter(make_item())
    )

    assert outputs[0].image.size == (100, 100)
    assert outputs[0].meta["character_crop_mode"] == "uncropped_single"


def test_min_area_ratio_crops_when_threshold_is_not_met(monkeypatch):
    monkeypatch.setattr(
        cropping,
        "detect_person",
        lambda *args, **kwargs: [detection((0, 0, 20, 25))],
    )

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            single_character_uncropped_min_area_ratio=0.10,
        ).iter(make_item())
    )

    assert outputs[0].image.size == (20, 25)
    assert outputs[0].meta["character_crop_mode"] == "person_crop"


def test_zero_min_area_ratio_disables_area_check(monkeypatch):
    monkeypatch.setattr(
        cropping,
        "detect_person",
        lambda *args, **kwargs: [detection((0, 0, 5, 5))],
    )

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            single_character_uncropped_min_area_ratio=0,
        ).iter(make_item())
    )

    assert outputs[0].image.size == (100, 100)


def test_min_crop_size_does_not_drop_preserved_original(monkeypatch):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])
    crop_action = SingleCharacterAwarePersonSplitAction(
        keep_single_character_uncropped=True,
        min_crop_size=320,
        person_conf={"level": "n"},
    )

    outputs = list(crop_action.iter(make_item((100, 100))))

    assert len(outputs) == 1
    assert outputs[0].image.size == (100, 100)


def test_crop_with_head_counts_only_valid_person_detections(monkeypatch):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])
    monkeypatch.setattr(cropping, "detect_heads", lambda *args, **kwargs: [])

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            crop_with_head=True,
        ).iter(make_item())
    )

    assert outputs == []


def test_crop_with_face_counts_only_valid_person_detections(monkeypatch):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])
    monkeypatch.setattr(cropping, "detect_faces", lambda *args, **kwargs: [])

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            crop_with_face=True,
        ).iter(make_item())
    )

    assert outputs == []


def test_three_stage_single_keeps_original_without_derived_crops(monkeypatch):
    calls = {"person": 0, "halfbody": 0, "head": 0}

    def detect_person_once(*args, **kwargs):
        calls["person"] += 1
        return [detection()]

    def unexpected_halfbody(*args, **kwargs):
        calls["halfbody"] += 1
        return [detection()]

    def unexpected_head(*args, **kwargs):
        calls["head"] += 1
        return [detection()]

    monkeypatch.setattr(cropping, "detect_person", detect_person_once)
    monkeypatch.setattr(cropping, "detect_halfbody", unexpected_halfbody)
    monkeypatch.setattr(cropping, "detect_heads", unexpected_head)

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            use_3stage_crop=True,
        ).iter(make_item())
    )

    assert len(outputs) == 1
    assert outputs[0].image.size == (100, 100)
    assert calls == {"person": 1, "halfbody": 0, "head": 0}


def test_three_stage_multi_keeps_existing_person_halfbody_head_shape(monkeypatch):
    calls = {"person": 0}

    def detect_people(*args, **kwargs):
        calls["person"] += 1
        return [
            detection((0, 0, 40, 100)),
            detection((50, 0, 100, 100)),
        ]

    monkeypatch.setattr(cropping, "detect_person", detect_people)
    monkeypatch.setattr(
        cropping,
        "detect_halfbody",
        lambda image, **kwargs: [((0, 0, image.width, image.height // 2), "halfbody", 0.8)],
    )
    monkeypatch.setattr(
        cropping,
        "detect_heads",
        lambda image, **kwargs: [((0, 0, image.width // 2, image.height // 2), "head", 0.8)],
    )

    outputs = list(
        action(
            keep_single_character_uncropped=True,
            use_3stage_crop=True,
        ).iter(make_item())
    )

    assert len(outputs) == 6
    assert calls["person"] == 1
    assert {output.meta["crop"]["type"] for output in outputs} == {
        "person",
        "halfbody",
        "head",
    }


def test_uncropped_single_can_be_loaded_by_stage3(monkeypatch, tmp_path):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])
    output = list(action(keep_single_character_uncropped=True).iter(make_item()))[0]
    output.meta["path"] = str(tmp_path / "raw" / "frame.png")
    output_dir = tmp_path / "cropped"
    exporter = SaveExporter(str(output_dir), no_meta=False, save_caption=False)
    exporter.pre_export()
    exporter.export_item(output)
    monkeypatch.setattr(
        file_utils,
        "ccip_extract_feature",
        lambda path: np.array([1.0, 2.0], dtype=np.float32),
    )

    image_files, features, _, _ = file_utils.load_image_features_and_characters(
        src_dir=str(output_dir),
        save_ccip_cache=False,
    )

    assert len(image_files) == 1
    assert features.shape == (1, 2)


def test_uncropped_single_is_reference_matched_into_character_folder(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])
    source_dir = tmp_path / "cropped"
    reference_dir = tmp_path / "ref" / "hero"
    classified_dir = tmp_path / "classified"
    reference_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(reference_dir / "reference.png")
    output = list(action(keep_single_character_uncropped=True).iter(make_item()))[0]
    exporter = SaveExporter(str(source_dir), no_meta=False, save_caption=False)
    exporter.pre_export()
    exporter.export_item(output)
    monkeypatch.setattr(
        file_utils,
        "ccip_extract_feature",
        lambda path: np.ones(8, dtype=np.float32),
    )

    def fake_cluster(images, **kwargs):
        count = len(images)
        return (
            np.zeros(count, dtype=int),
            np.zeros((count, count), dtype=np.float32),
            np.ones((count, count), dtype=bool),
        )

    monkeypatch.setattr(classification, "cluster_characters_basics", fake_cluster)
    monkeypatch.setattr(
        merge_clusters_module, "ccip_difference", lambda left, right: 0.0
    )

    counts = classify_from_directory(
        str(source_dir),
        str(classified_dir),
        ref_dir=str(tmp_path / "ref"),
        to_extract_from_noise=False,
        to_filter=False,
        keep_unnamed=False,
        clu_min_samples=2,
        move=False,
    )

    assert counts == {"hero": 1}
    assert (classified_dir / "hero" / "frame.png").is_file()


def write_metadata(image_path, metadata):
    meta_path, _ = get_corr_meta_names(str(image_path))
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


def test_stage4_does_not_duplicate_uncropped_stage2_original(tmp_path):
    raw_dir = tmp_path / "raw"
    classified_dir = tmp_path / "classified"
    output_dir = tmp_path / "training"
    raw_dir.mkdir()
    classified_dir.mkdir()
    (output_dir / "classified").mkdir(parents=True)
    raw_path = raw_dir / "frame.png"
    classified_path = classified_dir / "frame.png"
    Image.new("RGB", (100, 100), "white").save(raw_path)
    Image.new("RGB", (100, 100), "white").save(classified_path)
    raw_meta = {
        "filename": "frame.png",
        "path": str(raw_path.resolve()),
        "current_path": str(raw_path.resolve()),
        "image_size": [100, 100],
        "characters": ["hero"],
    }
    classified_meta = {
        **raw_meta,
        "current_path": str(classified_path.resolve()),
        "character_crop_mode": "uncropped_single",
        "person_count": 1,
    }
    write_metadata(raw_path, raw_meta)
    write_metadata(classified_path, classified_meta)

    resize_character_images(
        [str(classified_dir), str(raw_dir)],
        str(output_dir),
        max_size=768,
        ext=".png",
        image_type="screenshots",
        n_nocharacter_frames=0,
        to_resize=False,
    )

    assert len(list(output_dir.rglob("*.png"))) == 1


def test_stage2_summary_reports_all_decision_counters(caplog):
    stats = Stage2CropStats(
        source_images_processed=5,
        images_without_valid_character=1,
        single_character_images=2,
        single_character_originals_kept=1,
        single_character_images_cropped=1,
        multi_character_images=2,
        character_crops_generated=5,
    )

    with caplog.at_level(logging.INFO):
        log_stage2_crop_summary(logging.getLogger("stage2-test"), stats, True, 0.1)

    assert "Stage 2 - Character Detection Summary" in caplog.text
    assert "Single-character originals kept:      1" in caplog.text
    assert "Character crops generated:            5" in caplog.text
    assert "min_area_ratio:                  0.1" in caplog.text


def test_waifuc_action_copy_updates_shared_job_local_stats(monkeypatch, tmp_path):
    class OneImageSource(BaseDataSource):
        def _iter(self):
            yield make_item()

    monkeypatch.setattr(cropping, "detect_person", lambda *args, **kwargs: [detection()])
    stats = Stage2CropStats()
    crop_action = action(
        keep_single_character_uncropped=True,
        stats=stats,
    )

    OneImageSource().attach(crop_action).export(
        SaveExporter(str(tmp_path), no_meta=False, save_caption=False)
    )

    snapshot = stats.snapshot()
    assert snapshot["source_images_processed"] == 1
    assert snapshot["single_character_originals_kept"] == 1
