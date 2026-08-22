"""Sidecar recognizer for rare shikigami missing from the bundled oashya model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from oashya.labels import CLASSINDEX as CI

from module.logger import logger


@dataclass(frozen=True)
class RareTemplate:
    name: str
    image: np.ndarray
    mask: np.ndarray
    width: int
    height: int
    masked_std: float


class NewRareRecognizer:
    """Detect curated new rare sprites and return oashya-compatible tracks."""

    TRACK_ID_BASE = 900_000
    SEARCH_TOP = 140
    SEARCH_BOTTOM = 620
    SCALE = 0.25

    def __init__(self, asset_dir: Path | None = None):
        self.asset_dir = asset_dir or Path(__file__).resolve().parents[1] / "new_rare_templates"
        manifest_path = self.asset_dir / "manifest.json"
        if not manifest_path.exists():
            self.threshold = 1.1
            self.templates = []
            logger.warning(f"New rare templates not found: {manifest_path}")
            return

        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        self.threshold = float(manifest["threshold"])
        self.templates = [self._load_template(item) for item in manifest["templates"]]
        self.names = sorted({template.name for template in self.templates})
        self._visible_names: set[str] = set()

    def _load_template(self, item: dict) -> RareTemplate:
        rgba = cv2.imread(str(self.asset_dir / item["file"]), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Invalid new rare template: {item['file']}")

        width = max(1, round(rgba.shape[1] * self.SCALE))
        height = max(1, round(rgba.shape[0] * self.SCALE))
        color = cv2.resize(rgba[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(rgba[:, :, 3], (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        masked_pixels = image[mask > 0]
        return RareTemplate(
            name=item["name"],
            image=image,
            mask=mask,
            width=int(item["width"]),
            height=int(item["height"]),
            masked_std=float(np.std(masked_pixels)),
        )

    def _best_match(self, image: np.ndarray, template: RareTemplate) -> tuple[float, int, int]:
        result = cv2.matchTemplate(
            image,
            template.image,
            cv2.TM_CCOEFF_NORMED,
            mask=template.mask,
        )
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, location = cv2.minMaxLoc(result)
        x, y = location
        patch = image[y:y + template.image.shape[0], x:x + template.image.shape[1]]
        patch_std = float(np.std(patch[template.mask > 0]))
        if patch_std < max(8.0, template.masked_std * 0.4):
            return 0.0, x, y
        return float(score), location[0], location[1]

    def detect(self, image: np.ndarray) -> list[tuple]:
        if not self.templates or image.shape[:2] != (720, 1280):
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=self.SCALE, fy=self.SCALE, interpolation=cv2.INTER_AREA)
        top = round(self.SEARCH_TOP * self.SCALE)
        bottom = round(self.SEARCH_BOTTOM * self.SCALE)
        search_image = scaled[top:bottom]

        best_by_name: dict[str, tuple[float, int, int, RareTemplate]] = {}
        for template in self.templates:
            score, x, y = self._best_match(search_image, template)
            current = best_by_name.get(template.name)
            if current is None or score > current[0]:
                best_by_name[template.name] = (score, x, y, template)

        tracks = []
        visible_names = set()
        for name, (score, x, y, template) in best_by_name.items():
            if score < self.threshold:
                continue
            visible_names.add(name)
            x = round(x / self.SCALE)
            y = round((y + top) / self.SCALE)
            tracks.append(
                (
                    self.TRACK_ID_BASE + self.names.index(name),
                    CI.MIN_SP,
                    score,
                    x + template.width / 2,
                    y + template.height / 2,
                    template.width,
                    template.height,
                    -0.1,
                )
            )

        for name in sorted(visible_names - self._visible_names):
            logger.info(f"New rare shikigami detected: {name}")
        self._visible_names = visible_names
        return tracks

    __call__ = detect
