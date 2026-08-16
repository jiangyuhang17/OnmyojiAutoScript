"""Build review sheets for foreground objects missed by the bundled tracker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPTURE_DIR = ROOT / "log" / "hya" / "20260815T23"
DEFAULT_ANALYSIS_DIR = (
    ROOT / "tasks" / "Hyakkiyakou" / "dataset" / "new_shikigami" / "real_hyakki"
)


def load_tracks(index_path: Path) -> dict[str, list[dict]]:
    result = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        result[Path(record["source"]).name] = record["tracks"]
    return result


def intersection_over_component(component: tuple[int, int, int, int], track: dict) -> float:
    x, y, width, height = component
    cx, cy, track_width, track_height = track["xywh"]
    tx1 = cx - track_width / 2
    ty1 = cy - track_height / 2
    tx2 = cx + track_width / 2
    ty2 = cy + track_height / 2
    overlap_width = max(0, min(x + width, tx2) - max(x, tx1))
    overlap_height = max(0, min(y + height, ty2) - max(y, ty1))
    return overlap_width * overlap_height / max(1, width * height)


def foreground_components(image: np.ndarray, background: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    difference = cv2.absdiff(image, background)
    mask = (np.max(difference, axis=2) >= 34).astype(np.uint8) * 255
    mask[:170] = 0
    mask[610:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 650 or width < 35 or height < 55:
            continue
        if width > 650 or height > 430:
            continue
        components.append((int(x), int(y), int(width), int(height), int(area)))
    return components


def difference_hash(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)


def is_duplicate(signature: np.ndarray, signatures: list[np.ndarray]) -> bool:
    return any(np.mean(np.abs(signature - previous)) < 5.0 for previous in signatures[-80:])


def discover(capture_dir: Path, analysis_dir: Path, limit: int) -> list[dict]:
    background_path = analysis_dir / "background.jpg"
    if not background_path.exists():
        background_path = analysis_dir / "proposals" / "background.jpg"
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(background_path)
    tracks_by_image = load_tracks(analysis_dir / "candidates.jsonl")

    candidates = []
    signatures = []
    for path in sorted(capture_dir.glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tracks = tracks_by_image.get(path.name, [])
        for x, y, width, height, area in foreground_components(image, background):
            component = (x, y, width, height)
            if any(intersection_over_component(component, track) >= 0.42 for track in tracks):
                continue
            padding = 18
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + width + padding)
            y2 = min(image.shape[0], y + height + padding)
            crop = image[y1:y2, x1:x2]
            signature = difference_hash(crop)
            if is_duplicate(signature, signatures):
                continue
            signatures.append(signature)
            candidates.append(
                {
                    "source": path.name,
                    "bbox": [x, y, width, height],
                    "area": area,
                    "crop": crop,
                }
            )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates[:limit]


def write_sheets(candidates: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 320, 220
    columns, rows = 4, 5
    page_size = columns * rows
    index = []
    for candidate_index, candidate in enumerate(candidates):
        crop_name = f"candidate_{candidate_index:04d}.jpg"
        cv2.imwrite(str(output_dir / crop_name), candidate["crop"])
        index.append({key: value for key, value in candidate.items() if key != "crop"} | {"crop": crop_name})

    for page_start in range(0, len(candidates), page_size):
        canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
        for local_index, candidate in enumerate(candidates[page_start:page_start + page_size]):
            row, column = divmod(local_index, columns)
            crop = candidate["crop"]
            scale = min((tile_width - 8) / crop.shape[1], (tile_height - 30) / crop.shape[0])
            resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            x = column * tile_width + (tile_width - resized.shape[1]) // 2
            y = row * tile_height + 22 + (tile_height - 22 - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            label = f"{page_start + local_index:04d} {candidate['source'][4:-4]}"
            cv2.putText(
                canvas,
                label,
                (column * tile_width + 5, row * tile_height + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        page = page_start // page_size + 1
        cv2.imwrite(str(output_dir / f"sheet_{page:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {len(candidates)} candidates and {(len(candidates) + page_size - 1) // page_size} sheets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    output_dir = args.output_dir or args.analysis_dir / "unrecognized"
    write_sheets(discover(args.capture_dir, args.analysis_dir, args.limit), output_dir)


if __name__ == "__main__":
    main()
