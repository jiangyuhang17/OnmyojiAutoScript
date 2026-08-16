"""Build masked templates for new rare shikigami from real Hyakki captures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CAPTURE_DIR = ROOT / "log" / "hya" / "20260815T23"
OUTPUT_DIR = ROOT / "tasks" / "Hyakkiyakou" / "new_rare_templates"
PUBLIC_ROOT = (
    ROOT
    / "tasks"
    / "Hyakkiyakou"
    / "dataset"
    / "new_shikigami"
    / "public_hyakki"
    / "BV1uwgP69Eje_rewards_v3"
)
PUBLIC_V4_ROOT = PUBLIC_ROOT.with_name("BV1uwgP69Eje_rewards_v4")


@dataclass(frozen=True)
class Annotation:
    source: str
    filename: str
    bbox: tuple[int, int, int, int]
    include: tuple[tuple[int, int, int, int], ...] = ()


SOURCES = {
    "local": {
        "directory": CAPTURE_DIR,
        "background": None,
    },
    "local_20260816T17": {
        "directory": ROOT / "log" / "hya" / "20260816T17",
        "background": None,
    },
    "public_20260814": {
        "directory": PUBLIC_ROOT / "targets" / "ssr" / "taira_no_masakado",
        "background": PUBLIC_ROOT / "events" / "event_0277_07751.744.jpg",
    },
    "public_kannazuki": {
        "directory": (
            ROOT
            / "tasks"
            / "Hyakkiyakou"
            / "dataset"
            / "new_shikigami"
            / "public_hyakki"
            / "BV1w9ZvBjEGi_rewards_v3"
            / "targets"
            / "ssr"
            / "kannazuki"
        ),
        "background": (
            ROOT
            / "tasks"
            / "Hyakkiyakou"
            / "dataset"
            / "new_shikigami"
            / "public_hyakki"
            / "BV1w9ZvBjEGi_rewards_v3"
            / "targets"
            / "ssr"
            / "kannazuki"
            / "event_0004_00_00038.400.jpg"
        ),
    },
    "public_longjue": {
        "directory": PUBLIC_V4_ROOT / "longjue_pre",
        "background": None,
    },
    "public_yaoyin": {
        "directory": PUBLIC_V4_ROOT / "yaoyin_pre",
        "background": None,
    },
    "public_mengyin": {
        "directory": PUBLIC_V4_ROOT / "mengyin_1",
        "background": None,
    },
    "public_shiyue": {
        "directory": PUBLIC_V4_ROOT / "shiyue_pre",
        "background": None,
    },
}

# Coordinates are target-only regions in normalized 1280x720 captures.
ANNOTATIONS = {
    "arakuro": [
        Annotation("local", "hya_1786808142237.jpg", (745, 215, 990, 545)),
        Annotation("local", "hya_1786808143012.jpg", (585, 215, 890, 550)),
        Annotation("local", "hya_1786808143828.jpg", (425, 215, 745, 555)),
    ],
    "shiyao_takiyashahime": [
        Annotation("local", "hya_1786808359590.jpg", (775, 245, 1105, 570)),
        Annotation("local", "hya_1786808360090.jpg", (690, 245, 1000, 570)),
        Annotation("local", "hya_1786808360677.jpg", (610, 240, 900, 570)),
        Annotation("local", "hya_1786808361788.jpg", (425, 245, 710, 570)),
    ],
    "chenhui_ebisu": [
        Annotation("local", "hya_1786808440713.jpg", (795, 245, 1125, 585)),
        Annotation("local", "hya_1786808441216.jpg", (715, 245, 1045, 585)),
        Annotation("local", "hya_1786808442193.jpg", (545, 245, 900, 585)),
    ],
    "taira_no_masakado": [
        Annotation("local_20260816T17", "hya_1786871038237.jpg", (535, 215, 865, 565)),
        Annotation("local_20260816T17", "hya_1786871039075.jpg", (385, 215, 720, 565)),
        Annotation("local_20260816T17", "hya_1786871039898.jpg", (245, 215, 600, 570)),
        Annotation("local_20260816T17", "hya_1786871040684.jpg", (115, 215, 475, 570)),
        Annotation(
            "public_20260814",
            "event_0277_00_07748.405.jpg",
            (500, 270, 810, 565),
            ((600, 315, 800, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_01_07748.882.jpg",
            (445, 275, 735, 565),
            ((590, 315, 730, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_02_07749.359.jpg",
            (360, 275, 660, 565),
            ((450, 315, 650, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_03_07749.836.jpg",
            (250, 280, 555, 565),
            ((390, 315, 550, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_04_07750.313.jpg",
            (80, 270, 500, 565),
            ((240, 315, 480, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_05_07750.790.jpg",
            (100, 275, 445, 565),
            ((240, 315, 440, 455),),
        ),
    ],
    "kannazuki": [
        Annotation("local_20260816T17", "hya_1786871141282.jpg", (905, 220, 1190, 595)),
        Annotation("local_20260816T17", "hya_1786871141966.jpg", (715, 220, 1010, 595)),
        Annotation("local_20260816T17", "hya_1786871142755.jpg", (590, 220, 890, 595)),
        Annotation("local_20260816T17", "hya_1786871143453.jpg", (450, 220, 755, 595)),
        Annotation("local_20260816T17", "hya_1786871144166.jpg", (370, 220, 680, 595)),
        Annotation(
            "public_kannazuki",
            "event_0004_04_00039.467.jpg",
            (900, 285, 1180, 590),
            ((930, 325, 1160, 565),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_06_00040.000.jpg",
            (880, 285, 1155, 590),
            ((910, 325, 1140, 565),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_08_00040.533.jpg",
            (780, 285, 1060, 590),
            ((805, 325, 1045, 565),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_10_00041.067.jpg",
            (690, 285, 965, 585),
            ((755, 340, 950, 555),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_12_00041.600.jpg",
            (620, 290, 905, 590),
            ((650, 335, 890, 565),),
        ),
    ],
    "shenniang_hoshiguma_dojo": [
        Annotation("local_20260816T17", "hya_1786871239181.jpg", (290, 360, 465, 580)),
        Annotation("local_20260816T17", "hya_1786871239826.jpg", (175, 355, 350, 580)),
        Annotation("local_20260816T17", "hya_1786871240563.jpg", (45, 350, 220, 580)),
    ],
    "fengyangjun": [
        Annotation("local_20260816T17", "hya_1786871248295.jpg", (930, 330, 1280, 590)),
        Annotation("local_20260816T17", "hya_1786871248958.jpg", (825, 330, 1195, 590)),
        Annotation("local_20260816T17", "hya_1786871249620.jpg", (700, 330, 1085, 590)),
    ],
    "xinyou_inugami": [
        Annotation("local_20260816T17", "hya_1786871176114.jpg", (425, 365, 615, 575)),
        Annotation("local_20260816T17", "hya_1786871177388.jpg", (220, 365, 450, 580)),
        Annotation("local_20260816T17", "hya_1786871178077.jpg", (115, 365, 345, 580)),
    ],
    "longjue": [
        Annotation(
            "public_longjue",
            "frame_05621.500.jpg",
            (760, 245, 1080, 585),
            ((790, 270, 1060, 555),),
        ),
        Annotation(
            "public_longjue",
            "frame_05622.000.jpg",
            (670, 245, 1000, 585),
            ((710, 270, 965, 555),),
        ),
        Annotation(
            "public_longjue",
            "frame_05622.500.jpg",
            (575, 245, 905, 590),
            ((610, 270, 870, 560),),
        ),
        Annotation(
            "public_longjue",
            "frame_05623.000.jpg",
            (485, 245, 820, 595),
            ((520, 270, 785, 565),),
        ),
    ],
    "yaoyin_kinnara": [
        Annotation(
            "public_yaoyin",
            "frame_06058.500.jpg",
            (810, 245, 1140, 590),
            ((875, 270, 1115, 555),),
        ),
        Annotation(
            "public_yaoyin",
            "frame_06059.000.jpg",
            (700, 245, 1030, 590),
            ((760, 270, 1000, 555),),
        ),
        Annotation(
            "public_yaoyin",
            "frame_06061.000.jpg",
            (360, 245, 690, 590),
            ((405, 270, 630, 550),),
        ),
        Annotation(
            "public_yaoyin",
            "frame_06062.000.jpg",
            (175, 245, 500, 590),
            ((210, 270, 440, 550),),
        ),
    ],
    "mengyin_kocho": [
        Annotation(
            "public_mengyin",
            "frame_03458.000.jpg",
            (900, 250, 1230, 575),
            ((940, 280, 1200, 550),),
        ),
        Annotation(
            "public_mengyin",
            "frame_03458.500.jpg",
            (805, 250, 1135, 575),
            ((850, 280, 1105, 550),),
        ),
        Annotation(
            "public_mengyin",
            "frame_03459.000.jpg",
            (735, 250, 1060, 575),
            ((780, 280, 1030, 550),),
        ),
    ],
    "shiyue_kyuketsuhime": [
        Annotation(
            "public_shiyue",
            "frame_03472.000.jpg",
            (735, 285, 1060, 585),
            ((755, 300, 955, 570),),
        ),
        Annotation(
            "public_shiyue",
            "frame_03474.500.jpg",
            (410, 285, 735, 590),
            ((425, 300, 610, 570),),
        ),
        Annotation(
            "public_shiyue",
            "frame_03475.500.jpg",
            (285, 285, 610, 590),
            ((300, 300, 490, 570),),
        ),
        Annotation(
            "public_shiyue",
            "frame_03476.500.jpg",
            (205, 285, 525, 590),
            ((220, 300, 410, 570),),
        ),
    ],
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != (720, 1280):
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return image


def build_background(capture_dir: Path) -> np.ndarray:
    paths = sorted(capture_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No captures in {capture_dir}")

    # Sampling keeps memory bounded while still removing moving foreground objects.
    sample_paths = paths[:: max(1, len(paths) // 80)]
    stack = np.stack([read_image(path) for path in sample_paths])
    return np.median(stack, axis=0).astype(np.uint8)


def foreground_mask(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(image, background)
    mask = (np.max(difference, axis=2) >= 30).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backgrounds = {}
    for source, definition in SOURCES.items():
        background_path = definition["background"]
        backgrounds[source] = (
            read_image(background_path)
            if background_path is not None
            else build_background(definition["directory"])
        )
    manifest = {"version": 1, "threshold": 0.85, "templates": []}

    for name, annotations in ANNOTATIONS.items():
        for index, annotation in enumerate(annotations):
            source = SOURCES[annotation.source]
            filename = annotation.filename
            x1, y1, x2, y2 = annotation.bbox
            image = read_image(source["directory"] / filename)
            mask = foreground_mask(image, backgrounds[annotation.source])
            crop = image[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]

            if annotation.include:
                allowed = np.zeros_like(crop_mask)
                for ix1, iy1, ix2, iy2 in annotation.include:
                    allowed[max(0, iy1 - y1):iy2 - y1, max(0, ix1 - x1):ix2 - x1] = 255
                crop_mask = cv2.bitwise_and(crop_mask, allowed)

            # Ignore sparse background noise and the always-present player at the bottom.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(crop_mask)
            clean_mask = np.zeros_like(crop_mask)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] >= 80:
                    clean_mask[labels == component] = 255

            output_name = f"{name}_{index}.png"
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = clean_mask
            cv2.imwrite(str(OUTPUT_DIR / output_name), rgba)
            manifest["templates"].append(
                {
                    "name": name,
                    "file": output_name,
                    "source": f"{annotation.source}:{filename}",
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    print(f"Built {len(manifest['templates'])} templates in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
