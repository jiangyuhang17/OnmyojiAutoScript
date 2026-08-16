"""Create overviews and frame samples from public Hyakki gameplay recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, second: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = capture.read()
    return frame if ok else None


def overview(video: Path, output: Path, interval: float, start: float = 0.0, end: float = 0.0) -> None:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    end = end if end > start else duration
    seconds = np.arange(start, min(end, duration), interval)
    tile_width, tile_height = 320, 200
    columns = 5
    rows = 4
    output.mkdir(parents=True, exist_ok=True)

    for page_start in range(0, len(seconds), columns * rows):
        canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
        for index, second in enumerate(seconds[page_start:page_start + columns * rows]):
            frame = read_frame(capture, float(second))
            if frame is None:
                continue
            scale = min(tile_width / frame.shape[1], (tile_height - 22) / frame.shape[0])
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            row, column = divmod(index, columns)
            x = column * tile_width + (tile_width - resized.shape[1]) // 2
            y = row * tile_height + 22 + (tile_height - 22 - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            label = f"{int(second) // 60:02d}:{int(second) % 60:02d}"
            cv2.putText(canvas, label, (column * tile_width + 5, row * tile_height + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        page = page_start // (columns * rows) + 1
        cv2.imwrite(str(output / f"overview_{page:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    capture.release()


def extract(video: Path, output: Path, start: float, end: float, interval: float, width: int) -> None:
    capture = cv2.VideoCapture(str(video))
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for second in np.arange(start, end, interval):
        frame = read_frame(capture, float(second))
        if frame is None:
            continue
        if width > 0 and frame.shape[1] != width:
            height = round(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output / f"frame_{second:09.3f}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        count += 1
    capture.release()
    print(f"Extracted {count} frames from {start:.1f}s to {end:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("overview", "extract"))
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    if args.action == "overview":
        overview(args.video, args.output, args.interval, args.start, args.end)
    else:
        if args.end <= args.start:
            raise SystemExit("--end must be greater than --start")
        extract(args.video, args.output, args.start, args.end, args.interval, args.width)


if __name__ == "__main__":
    main()
