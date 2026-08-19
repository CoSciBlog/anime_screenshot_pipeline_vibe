import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from imgutils.detect import detect_faces, detect_halfbody, detect_heads, detect_person
from waifuc.action.base import BaseAction
from waifuc.model import ImageItem


Detection = Tuple[Tuple[float, float, float, float], str, float]


@dataclass
class Stage2CropStats:
    """Job-local counters for the optional single-character crop mode."""

    source_images_processed: int = 0
    images_without_valid_character: int = 0
    single_character_images: int = 0
    single_character_originals_kept: int = 0
    single_character_images_cropped: int = 0
    multi_character_images: int = 0
    character_crops_generated: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def __deepcopy__(self, memo):
        # Waifuc deep-copies actions before execution. Sharing this job-local
        # counter lets the caller report the results after export finishes.
        return self

    def increment(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "source_images_processed": self.source_images_processed,
                "images_without_valid_character": self.images_without_valid_character,
                "single_character_images": self.single_character_images,
                "single_character_originals_kept": self.single_character_originals_kept,
                "single_character_images_cropped": self.single_character_images_cropped,
                "multi_character_images": self.multi_character_images,
                "character_crops_generated": self.character_crops_generated,
            }


@dataclass(frozen=True)
class _ValidPersonDetection:
    original_index: int
    area: Tuple[float, float, float, float]
    type_name: str
    score: float
    image: ImageItem


class SingleCharacterAwarePersonSplitAction(BaseAction):
    """Split valid people while optionally preserving a single-person source image.

    Person detection is executed exactly once per input image. Head and face
    requirements are evaluated on each detected person crop before the number of
    valid characters is used for the keep/crop decision.
    """

    def __init__(
        self,
        keep_single_character_uncropped: bool = False,
        single_character_uncropped_min_area_ratio: float = 0.0,
        min_crop_size: int = 320,
        crop_with_head: bool = False,
        crop_with_face: bool = False,
        use_3stage_crop: bool = False,
        person_conf: Optional[dict] = None,
        halfbody_conf: Optional[dict] = None,
        head_conf: Optional[dict] = None,
        keep_origin_tags: bool = False,
        logger: Optional[logging.Logger] = None,
        stats: Optional[Stage2CropStats] = None,
    ):
        ratio = float(single_character_uncropped_min_area_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "single_character_uncropped_min_area_ratio must be between 0 and 1"
            )
        if min_crop_size < 0:
            raise ValueError("min_crop_size must be non-negative")

        self.keep_single_character_uncropped = keep_single_character_uncropped
        self.single_character_uncropped_min_area_ratio = ratio
        self.min_crop_size = min_crop_size
        self.crop_with_head = crop_with_head
        self.crop_with_face = crop_with_face
        self.use_3stage_crop = use_3stage_crop
        self.person_conf = dict(person_conf or {})
        self.halfbody_conf = dict(halfbody_conf or {})
        self.head_conf = dict(head_conf or {})
        self.keep_origin_tags = keep_origin_tags
        self.logger = logger or logging.getLogger()
        self.stats = stats or Stage2CropStats()

    @staticmethod
    def _filename_parts(item: ImageItem):
        filename = item.meta.get("filename")
        return os.path.splitext(filename) if filename else (None, None)

    @staticmethod
    def _serializable_bbox(area) -> List[float]:
        return [float(value) for value in area]

    @staticmethod
    def _bbox_area_ratio(area, image: ImageItem) -> float:
        x0, y0, x1, y1 = area
        bbox_area = max(0.0, float(x1) - float(x0)) * max(
            0.0, float(y1) - float(y0)
        )
        image_area = image.image.width * image.image.height
        return min(1.0, bbox_area / image_area) if image_area else 0.0

    def _passes_identity_filters(self, item: ImageItem) -> bool:
        if self.crop_with_head:
            heads = detect_heads(
                item.image,
                "n",
                conf_threshold=0.3,
                iou_threshold=0.7,
            )
            if not heads:
                return False
        if self.crop_with_face:
            faces = detect_faces(
                item.image,
                "n",
                "v1.4",
                conf_threshold=0.25,
                iou_threshold=0.7,
            )
            if not faces:
                return False
        return True

    def _passes_output_filters(self, item: ImageItem) -> bool:
        if min(item.image.width, item.image.height) < self.min_crop_size:
            return False
        return self._passes_identity_filters(item)

    def _person_meta(
        self,
        item: ImageItem,
        detection: _ValidPersonDetection,
        valid_person_count: int,
        filename: Optional[str],
    ) -> dict:
        meta = {
            **item.meta,
            "crop": {
                "type": detection.type_name,
                "score": detection.score,
            },
            "character_crop_mode": "person_crop",
            "person_count": valid_person_count,
            "person_bbox": self._serializable_bbox(detection.area),
        }
        if "tags" in meta and not self.keep_origin_tags:
            del meta["tags"]
        if filename is not None:
            meta["filename"] = filename
        return meta

    def _derived_meta(
        self,
        item: ImageItem,
        detection: _ValidPersonDetection,
        valid_person_count: int,
        crop_type: str,
        crop_score: float,
        filename: Optional[str],
    ) -> dict:
        meta = {
            **item.meta,
            "crop": {"type": crop_type, "score": float(crop_score)},
            "character_crop_mode": "person_crop",
            "person_count": valid_person_count,
            "person_bbox": self._serializable_bbox(detection.area),
        }
        if "tags" in meta and not self.keep_origin_tags:
            del meta["tags"]
        if filename is not None:
            meta["filename"] = filename
        return meta

    def _valid_person_detections(
        self, item: ImageItem, detections
    ) -> List[_ValidPersonDetection]:
        valid = []
        for index, (area, type_name, score) in enumerate(detections):
            person_item = ImageItem(item.image.crop(area), dict(item.meta))
            if self._passes_identity_filters(person_item):
                valid.append(
                    _ValidPersonDetection(
                        original_index=index,
                        area=tuple(area),
                        type_name=str(type_name),
                        score=float(score),
                        image=person_item,
                    )
                )
        return valid

    def _iter_person_outputs(
        self,
        item: ImageItem,
        detections: List[_ValidPersonDetection],
    ) -> Iterator[ImageItem]:
        filebody, ext = self._filename_parts(item)
        valid_person_count = len(detections)
        for detection in detections:
            output_index = (
                detection.original_index + 1
                if self.use_3stage_crop
                else detection.original_index
            )
            person_filename = (
                f"{filebody}_person{output_index}{ext}"
                if filebody is not None
                else None
            )
            person_item = ImageItem(
                detection.image.image,
                self._person_meta(
                    item,
                    detection,
                    valid_person_count,
                    person_filename,
                ),
            )
            # Identity filters already ran on this exact person crop.
            if min(person_item.image.size) >= self.min_crop_size:
                self.stats.increment("character_crops_generated")
                yield person_item

            if not self.use_3stage_crop:
                continue

            half_detects = detect_halfbody(
                detection.image.image, **self.halfbody_conf
            )
            if half_detects:
                half_area, half_type, half_score = half_detects[0]
                half_filename = (
                    f"{filebody}_person{output_index}_halfbody{ext}"
                    if filebody is not None
                    else None
                )
                half_item = ImageItem(
                    detection.image.image.crop(half_area),
                    self._derived_meta(
                        item,
                        detection,
                        valid_person_count,
                        str(half_type),
                        float(half_score),
                        half_filename,
                    ),
                )
                if self._passes_output_filters(half_item):
                    self.stats.increment("character_crops_generated")
                    yield half_item

            head_detects = detect_heads(detection.image.image, **self.head_conf)
            if head_detects:
                (hx0, hy0, hx1, hy1), head_type, head_score = head_detects[0]
                cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
                width = height = max(hx1 - hx0, hy1 - hy0) * 1.5
                x0 = int(max(cx - width / 2, 0))
                y0 = int(max(cy - height / 2, 0))
                x1 = int(min(cx + width / 2, detection.image.image.width))
                y1 = int(min(cy + height / 2, detection.image.image.height))
                head_filename = (
                    f"{filebody}_person{output_index}_head{ext}"
                    if filebody is not None
                    else None
                )
                head_item = ImageItem(
                    detection.image.image.crop((x0, y0, x1, y1)),
                    self._derived_meta(
                        item,
                        detection,
                        valid_person_count,
                        str(head_type),
                        float(head_score),
                        head_filename,
                    ),
                )
                if self._passes_output_filters(head_item):
                    self.stats.increment("character_crops_generated")
                    yield head_item

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        self.stats.increment("source_images_processed")
        detections = detect_person(item.image, **self.person_conf)
        valid_detections = self._valid_person_detections(item, detections)
        person_count = len(valid_detections)

        if person_count == 0:
            self.stats.increment("images_without_valid_character")
            return

        if person_count == 1:
            self.stats.increment("single_character_images")
            detection = valid_detections[0]
            area_ratio = self._bbox_area_ratio(detection.area, item)
            threshold_met = (
                self.single_character_uncropped_min_area_ratio == 0
                or area_ratio >= self.single_character_uncropped_min_area_ratio
            )
            if self.keep_single_character_uncropped and threshold_met:
                meta = {
                    **item.meta,
                    "character_crop_mode": "uncropped_single",
                    "person_count": 1,
                    "person_bbox": self._serializable_bbox(detection.area),
                    "person_detection": {
                        "type": detection.type_name,
                        "score": detection.score,
                    },
                }
                self.stats.increment("single_character_originals_kept")
                self.logger.debug(
                    "Single character detected - keeping original: %s "
                    "bbox_area_ratio=%.4f",
                    item.meta.get("current_path", item.meta.get("filename", "<unknown>")),
                    area_ratio,
                )
                # min_crop_size intentionally does not apply to an uncropped source.
                yield ImageItem(item.image, meta)
                return

            self.stats.increment("single_character_images_cropped")
            if self.keep_single_character_uncropped:
                self.logger.debug(
                    "Single character detected but bbox too small - cropping: %s "
                    "bbox_area_ratio=%.4f required=%.4f",
                    item.meta.get("current_path", item.meta.get("filename", "<unknown>")),
                    area_ratio,
                    self.single_character_uncropped_min_area_ratio,
                )
        else:
            self.stats.increment("multi_character_images")

        yield from self._iter_person_outputs(item, valid_detections)

    def reset(self):
        # Statistics are intentionally retained in the shared, job-local object.
        pass


def log_stage2_crop_summary(
    logger: logging.Logger,
    stats: Stage2CropStats,
    keep_single_character_uncropped: bool,
    min_area_ratio: float,
) -> None:
    values = stats.snapshot()
    logger.info("=" * 60)
    logger.info("[SUMMARY] Stage 2 - Character Detection Summary")
    logger.info("=" * 60)
    logger.info(
        "Source images processed:              %d",
        values["source_images_processed"],
    )
    logger.info(
        "Images with no valid character:       %d",
        values["images_without_valid_character"],
    )
    logger.info(
        "Single-character images:              %d",
        values["single_character_images"],
    )
    logger.info(
        "Single-character originals kept:      %d",
        values["single_character_originals_kept"],
    )
    logger.info(
        "Single-character images cropped:      %d",
        values["single_character_images_cropped"],
    )
    logger.info(
        "Multi-character images:               %d",
        values["multi_character_images"],
    )
    logger.info(
        "Character crops generated:            %d",
        values["character_crops_generated"],
    )
    logger.info(
        "keep_single_character_uncropped: %s",
        str(bool(keep_single_character_uncropped)).lower(),
    )
    logger.info("min_area_ratio:                  %.4g", min_area_ratio)
    logger.info("=" * 60)
