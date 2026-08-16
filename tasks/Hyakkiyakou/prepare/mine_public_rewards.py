"""Mine named shikigami samples from reward banners in public Hyakki videos."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from module.ocr.models import get_ocr_model


ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = ROOT / "tasks" / "Hyakkiyakou" / "prepare" / "new_shikigami_sources.json"
BANNER = (390, 115, 890, 205)
TEXT_BANNER = (450, 130, 850, 190)


@dataclass
class RewardEvent:
    second: float
    context: list[tuple[float, np.ndarray]]
    banner_frame: np.ndarray | None = None
    banner_score: float = 0.0
    signature: np.ndarray | None = None


def banner_score(frame: np.ndarray) -> float:
    x1, y1, x2, y2 = BANNER
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 120)))


def ocr_banner(frame: np.ndarray, model) -> tuple[str, float]:
    x1, y1, x2, y2 = TEXT_BANNER
    text, score = model.ocr_single_line(frame[y1:y2, x1:x2])
    return text, float(score)


def text_signature(frame: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = TEXT_BANNER
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (100, 15), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_targets() -> list[dict]:
    data = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    return [
        {"slug": entry["slug"], "name": entry["name"], "rarity": entry["rarity"]}
        for entry in data["entries"]
    ]


def reward_name(text: str) -> str:
    value = text.replace(" ", "")
    if "获得" in value:
        value = value.split("获得", 1)[1]
    elif "得" in value[:3]:
        value = value.split("得", 1)[1]
    for marker in ("契约", "碎片", "*", "！", "!"):
        if marker in value:
            value = value.split(marker, 1)[0]
    return value


def match_target(text: str, targets: list[dict]) -> tuple[dict | None, float]:
    candidate = reward_name(text)
    best_target = None
    best_score = 0.0
    for target in targets:
        name = target["name"]
        if name in text or name == candidate:
            score = 1.0
        elif len(name) <= 2 or len(candidate) < len(name) - 1:
            score = 0.0
        else:
            score = SequenceMatcher(None, name, candidate).ratio()
        if score > best_score:
            best_target = target
            best_score = score
    if best_score < 0.80:
        return None, best_score
    return best_target, best_score


def normalized(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == (720, 1280):
        return frame
    return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)


def prepare_frame(frame: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is not None:
        x, y, width, height = crop
        frame = frame[y:y + height, x:x + width]
    return normalized(frame)


def save_event(event: RewardEvent, index: int, output: Path, targets: list[dict], model) -> dict:
    text, ocr_score = ocr_banner(event.banner_frame, model) if event.banner_frame is not None else ("", 0.0)
    target, match_score = match_target(text, targets)
    event_dir = output / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    if event.banner_frame is not None:
        cv2.imwrite(
            str(event_dir / f"event_{index:04d}_{event.second:09.3f}.jpg"),
            normalized(event.banner_frame),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    if target is not None:
        target_dir = output / "targets" / target["rarity"].lower() / target["slug"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for context_index, (second, frame) in enumerate(event.context):
            cv2.imwrite(
                str(target_dir / f"event_{index:04d}_{context_index:02d}_{second:09.3f}.jpg"),
                normalized(frame),
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )

    return {
        "index": index,
        "second": round(event.second, 3),
        "text": text,
        "reward_name": reward_name(text),
        "ocr_score": round(ocr_score, 4),
        "target": target["slug"] if target else None,
        "target_name": target["name"] if target else None,
        "match_score": round(match_score, 4),
    }


def mine(
    video: Path,
    output: Path,
    interval: float,
    threshold: float,
    crop: tuple[int, int, int, int] | None,
) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    sample_step = max(1, round(fps * interval))
    context = deque(maxlen=max(1, round(4.0 / interval)))
    targets = load_targets()
    model = get_ocr_model("ch")
    events = []
    active: RewardEvent | None = None
    frame_index = 0
    next_sample = 0
    next_progress = 600.0

    while capture.grab():
        if frame_index < next_sample:
            frame_index += 1
            continue
        ok, frame = capture.retrieve()
        if not ok:
            break
        frame = prepare_frame(frame, crop)
        second = frame_index / fps
        context.append((second, frame.copy()))
        score = banner_score(frame)
        if score >= threshold:
            signature = text_signature(frame)
            if active is None:
                active = RewardEvent(
                    second=second,
                    context=list(context),
                    banner_frame=frame.copy(),
                    banner_score=score,
                    signature=signature,
                )
            elif float(np.mean(np.abs(signature - active.signature))) >= 10.0:
                events.append(save_event(active, len(events), output, targets, model))
                active = RewardEvent(
                    second=second,
                    context=list(context),
                    banner_frame=frame.copy(),
                    banner_score=score,
                    signature=signature,
                )
            elif score > active.banner_score:
                active.banner_frame = frame.copy()
                active.banner_score = score
                active.signature = signature
        elif active is not None:
            events.append(save_event(active, len(events), output, targets, model))
            active = None

        if second >= next_progress:
            print(f"[scan] {second / 60:.0f}/{duration / 60:.0f} min, {len(events)} rewards")
            next_progress += 600.0
        next_sample += sample_step
        frame_index += 1

    if active is not None:
        events.append(save_event(active, len(events), output, targets, model))
    capture.release()
    output.mkdir(parents=True, exist_ok=True)
    (output / "rewards.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    matched = [event for event in events if event["target"]]
    print(f"[done] {len(events)} rewards, {len(matched)} target matches: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--banner-threshold", type=float, default=0.65)
    parser.add_argument(
        "--crop",
        help="Optional game viewport as x,y,width,height before normalization",
    )
    args = parser.parse_args()
    crop = tuple(int(value) for value in args.crop.split(",")) if args.crop else None
    if crop is not None and len(crop) != 4:
        raise SystemExit("--crop requires x,y,width,height")
    mine(args.video, args.output, args.interval, args.banner_threshold, crop)


if __name__ == "__main__":
    main()
