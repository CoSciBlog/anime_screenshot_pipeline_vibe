import logging
import importlib

import numpy as np
from PIL import Image

from anime2sd.character import Character
from anime2sd.classif.file_utils import (
    get_episode_key,
    limit_recognized_character_images,
    save_to_dir,
)
from automatic_pipeline import (
    cleanup_classified_aux_files,
    cleanup_stage2_crops_after_classification,
)

classification = importlib.import_module("anime2sd.classif.classify_characters")


def test_episode_key_uses_episode_folder_or_file_pattern():
    assert get_episode_key(r"C:\input\S01E02\crop.png") == "S01E02"
    assert get_episode_key(r"C:\input\scene_s02e03_person0.png") == "S02E03"


def test_reference_character_limits_keep_noise_and_cap_repeated_episodes():
    image_files = np.array(
        [
            r"C:\input\S01E01\a.png",
            r"C:\input\S01E01\b.png",
            r"C:\input\S01E02\c.png",
            r"C:\input\S01E02\d.png",
            r"C:\input\S01E01\noise.png",
        ]
    )
    images = np.arange(5).reshape(5, 1)
    labels = np.array([0, 0, 0, 0, -1])
    mapping = {0: Character.from_string("frieren")}

    limited_files, limited_images, limited_labels = limit_recognized_character_images(
        image_files,
        images,
        labels,
        mapping,
        max_per_character=3,
        max_per_episode=1,
    )

    assert limited_files.tolist() == [
        r"C:\input\S01E01\a.png",
        r"C:\input\S01E02\c.png",
        r"C:\input\S01E01\noise.png",
    ]
    assert limited_images.tolist() == [[0], [2], [4]]
    assert limited_labels.tolist() == [0, 0, -1]


def test_cleanup_classified_aux_files_keeps_images(tmp_path):
    output = tmp_path / "classified" / "frieren"
    output.mkdir(parents=True)
    (output / "keep.png").write_text("image", encoding="utf-8")
    metadata = output / "metadata"
    metadata.mkdir()
    (metadata / "remove_meta.json").write_text("{}", encoding="utf-8")
    (metadata / "remove_ccip.npy").write_bytes(b"feature")

    cleanup_classified_aux_files(str(tmp_path / "classified"), logging.getLogger())

    assert (output / "keep.png").exists()
    assert not (metadata / "remove_meta.json").exists()
    assert not (metadata / "remove_ccip.npy").exists()


def test_cleanup_stage2_crops_removes_only_generated_classification_input(tmp_path):
    generated_crops = tmp_path / "dst" / "intermediate" / "screenshots" / "cropped"
    generated_crops.mkdir(parents=True)
    (generated_crops / "crop.png").write_text("crop", encoding="utf-8")
    source_crops = tmp_path / "user_input"
    source_crops.mkdir()
    (source_crops / "keep.png").write_text("source", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "dst_dir": str(tmp_path / "dst"),
            "extra_path_component": "",
            "image_type": "screenshots",
        },
    )()

    cleanup_stage2_crops_after_classification(args, str(source_crops), logging.getLogger())
    assert generated_crops.exists()
    assert (source_crops / "keep.png").exists()

    cleanup_stage2_crops_after_classification(args, str(generated_crops), logging.getLogger())
    assert not generated_crops.exists()
    assert (source_crops / "keep.png").exists()


def test_classified_output_stores_json_and_npy_in_metadata_subfolder(tmp_path):
    source = tmp_path / "cropped" / "frame.png"
    source.parent.mkdir()
    Image.new("RGB", (16, 16), color="white").save(source)
    output = tmp_path / "classified"

    save_to_dir(
        np.array([str(source)]),
        np.array([[0.25, 0.5]]),
        str(output),
        np.array([0]),
        {0: Character.from_string("frieren")},
    )

    character_dir = output / "frieren"
    assert (character_dir / "frame.png").exists()
    assert (character_dir / "metadata" / ".frame_meta.json").exists()
    assert (character_dir / "metadata" / ".frame_ccip.npy").exists()
    assert not (character_dir / ".frame_meta.json").exists()
    assert not (character_dir / ".frame_ccip.npy").exists()


def test_large_classification_chunks_quadratic_similarity_work(tmp_path, monkeypatch):
    image_files = np.array([str(tmp_path / f"frame_{index}.png") for index in range(10)])
    images = np.arange(20).reshape(10, 2)
    chunk_lengths = []
    saved_labels = []

    monkeypatch.setattr(
        classification,
        "load_image_features_and_characters",
        lambda *args, **kwargs: (image_files, images, None, {}),
    )

    def classify_batch(files, *_args):
        chunk_lengths.append(len(files))
        return np.zeros(len(files), dtype=int), {}

    monkeypatch.setattr(classification, "_classify_feature_batch", classify_batch)
    monkeypatch.setattr(classification, "remove_empty_folders", lambda _path: None)
    monkeypatch.setattr(
        classification,
        "save_to_dir",
        lambda _files, _images, _dst, labels, *_args, **_kwargs: saved_labels.extend(labels),
    )

    classification.classify_from_directory(
        str(tmp_path),
        str(tmp_path / "classified"),
        to_extract_from_noise=False,
        to_filter=False,
        clu_min_samples=2,
        classification_chunk_size=4,
    )

    assert chunk_lengths == [4, 4, 2]
    assert saved_labels == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
