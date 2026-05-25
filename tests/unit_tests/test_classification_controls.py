import logging

import numpy as np

from anime2sd.character import Character
from anime2sd.classif.file_utils import (
    get_episode_key,
    limit_recognized_character_images,
)
from automatic_pipeline import cleanup_classified_aux_files


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
    (output / "remove_meta.json").write_text("{}", encoding="utf-8")
    (output / "remove_ccip.npy").write_bytes(b"feature")

    cleanup_classified_aux_files(str(tmp_path / "classified"), logging.getLogger())

    assert (output / "keep.png").exists()
    assert not (output / "remove_meta.json").exists()
    assert not (output / "remove_ccip.npy").exists()
