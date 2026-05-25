import logging

import pytest
from PIL import Image

from anime2sd.basics import get_corr_ccip_names, get_corr_meta_names, parse_anime_info
from anime2sd.common_preprocess import rearrange_related_files


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "[SubsPlease] 16bit Sensation - Another Layer - 07 (1080p) [771BDD0C].mkv",
            ("16bit Sensation - Another Layer", 7),
        ),
        (
            "[HorribleSubs] Toaru Kagaku no Railgun T - 25 [1080p].mkv",
            ("Toaru Kagaku no Railgun T", 25),
        ),
        ("[Hayaisubs] Yama no Susume 2 - 18 [720p].mkv", ("Yama no Susume 2", 18)),
        # Add more test cases as needed
        (
            "[RandomGroup] Anime Title - Extra Info - 10 [720p].mkv",
            ("Anime Title - Extra Info", 10),
        ),
        (
            "Yama no Susume (Saison 2) 16 vostfr [720p]",
            ("Yama no Susume (Saison 2) 16 vostfr", None),
        ),
        (
            "[Ohys-Raws] Toaru Kagaku no Railgun T - SP2 (BD 1280x720 x264 AAC).mp4",
            ("Toaru Kagaku no Railgun T", None),
        ),
        (
            "[EA]Toaru_Kagaku_no_Railgun_T_24_[1920x1080][Hi10p][373BAEBF].mkv",
            ("Toaru_Kagaku_no_Railgun_T_24_", None),
        ),
        ("Only Title.mkv", ("Only Title", None)),
        ("[Group] Only Title - No Episode.mkv", ("Only Title", None)),
    ],
)
def test_parse_anime_info(filename, expected):
    assert parse_anime_info(filename) == expected


def test_rearrange_related_files_reports_generated_metadata_once(tmp_path, caplog):
    Image.new("RGB", (16, 16), color="white").save(tmp_path / "frame.png")

    with caplog.at_level(logging.INFO):
        rearrange_related_files(str(tmp_path))

    assert (tmp_path / "metadata" / ".frame_meta.json").exists()
    assert "Created default metadata for 1 image(s)" in caplog.text
    assert "No related file found" not in caplog.text


def test_related_sidecar_paths_use_metadata_subfolder(tmp_path):
    img_path = tmp_path / "frieren" / "frame.png"

    meta_path, _ = get_corr_meta_names(str(img_path))
    ccip_path, _ = get_corr_ccip_names(str(img_path))

    assert meta_path == str(tmp_path / "frieren" / "metadata" / ".frame_meta.json")
    assert ccip_path == str(tmp_path / "frieren" / "metadata" / ".frame_ccip.npy")


def test_rearrange_related_files_moves_legacy_sidecars_to_metadata_folder(tmp_path):
    Image.new("RGB", (16, 16), color="white").save(tmp_path / "frame.png")
    legacy_meta = tmp_path / ".frame_meta.json"
    legacy_meta.write_text("{}", encoding="utf-8")
    legacy_ccip = tmp_path / ".frame_ccip.npy"
    legacy_ccip.write_bytes(b"feature")

    rearrange_related_files(str(tmp_path))

    assert not legacy_meta.exists()
    assert not legacy_ccip.exists()
    assert (tmp_path / "metadata" / ".frame_meta.json").exists()
    assert (tmp_path / "metadata" / ".frame_ccip.npy").exists()
