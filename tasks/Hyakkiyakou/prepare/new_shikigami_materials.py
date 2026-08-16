import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'new_shikigami_sources.json'
DEFAULT_ROOT = HERE.parent / 'dataset' / 'new_shikigami'
VIDEO_SUFFIXES = {'.mp4', '.mkv', '.webm', '.mov'}


def load_entries(names: list[str]) -> list[dict]:
    with MANIFEST.open('r', encoding='utf-8') as file:
        entries = json.load(file)['entries']
    if not names:
        return entries
    wanted = {name.casefold() for name in names}
    selected = [
        entry for entry in entries
        if entry['slug'].casefold() in wanted or entry['name'].casefold() in wanted
    ]
    missing = wanted - {
        value.casefold()
        for entry in selected
        for value in (entry['slug'], entry['name'])
    }
    if missing:
        raise SystemExit(f'Unknown shikigami: {", ".join(sorted(missing))}')
    return selected


def entry_folder(root: Path, entry: dict) -> Path:
    return root / 'videos' / entry['rarity'].lower() / entry['slug']


def video_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)


def download(entries: list[dict], root: Path, cookies_from_browser: str | None) -> int:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise SystemExit('Install downloader with: python -m pip install yt-dlp')

    failures = 0
    archive = root / 'download.archive'
    archive.parent.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        folder = entry_folder(root, entry)
        if video_files(folder):
            print(f'[skip] {entry["rarity"]} {entry["name"]}: video exists')
            continue
        if not entry['sources']:
            print(f'[missing-source] {entry["rarity"]} {entry["name"]}')
            failures += 1
            continue

        folder.mkdir(parents=True, exist_ok=True)
        downloaded = False
        for source_index, source in enumerate(entry['sources'], start=1):
            output = folder / f'{source_index:02d}_{source["kind"]}_%(id)s.%(ext)s'
            command = [
                sys.executable, '-m', 'yt_dlp',
                '--no-overwrites',
                '--download-archive', str(archive),
                '--write-info-json',
                '--format', 'best[height<=720][vcodec^=avc1]/best[height<=720]/best',
                '--output', str(output),
            ]
            if cookies_from_browser:
                command.extend(['--cookies-from-browser', cookies_from_browser])
            command.append(source['url'])
            print(f'[download] {entry["rarity"]} {entry["name"]}: {source["url"]}')
            result = subprocess.run(command, check=False)
            if result.returncode == 0 and video_files(folder):
                downloaded = True
                break
        if not downloaded:
            print(f'[failed] {entry["rarity"]} {entry["name"]}')
            failures += 1
    return failures


def frame_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)


def extract_video(video: Path, destination: Path, interval: float, threshold: float) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f'[invalid-video] {video}')
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        print(f'[invalid-metadata] {video}')
        return 0

    start_frame = int(frame_count * 0.08)
    end_frame = int(frame_count * 0.92)
    step = max(1, round(fps * interval))
    destination.mkdir(parents=True, exist_ok=True)
    previous = None
    saved = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    next_sample = start_frame
    while frame_index < end_frame and capture.grab():
        if frame_index >= next_sample:
            ok, frame = capture.retrieve()
            if ok:
                signature = frame_signature(frame)
                difference = np.inf if previous is None else np.mean(np.abs(signature - previous))
                if difference >= threshold:
                    previous = signature
                    output = destination / f'{video.stem}_{frame_index:07d}.jpg'
                    if cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        saved += 1
            next_sample += step
        frame_index += 1
    capture.release()
    return saved


def extract(entries: list[dict], root: Path, interval: float, threshold: float) -> int:
    total = 0
    for entry in entries:
        folder = entry_folder(root, entry)
        for video in video_files(folder):
            destination = root / 'frames' / entry['rarity'].lower() / entry['slug'] / video.stem
            count = extract_video(video, destination, interval, threshold)
            total += count
            print(f'[extract] {entry["rarity"]} {entry["name"]}: {count} frames')
    print(f'[extract-total] {total} frames')
    return 0


def status(entries: list[dict], root: Path) -> int:
    source_count = sum(bool(entry['sources']) for entry in entries)
    video_count = 0
    frame_count = 0
    for entry in entries:
        videos = video_files(entry_folder(root, entry))
        frames = list((root / 'frames' / entry['rarity'].lower() / entry['slug']).rglob('*.jpg'))
        video_count += bool(videos)
        frame_count += len(frames)
        state = 'frames' if frames else 'video' if videos else 'source' if entry['sources'] else 'missing'
        print(f'[{state:7}] {entry["rarity"]:3} {entry["name"]}: {len(videos)} video, {len(frames)} frames')
    print(f'[summary] {len(entries)} classes, {source_count} sourced, {video_count} downloaded, {frame_count} frames')
    return 0


def overview(entries: list[dict], root: Path) -> int:
    tile_width = 600
    tile_height = 169
    columns = 2
    tiles = []
    for entry in entries:
        files = sorted(
            (root / 'frames' / entry['rarity'].lower() / entry['slug']).rglob('*.jpg')
        )
        if not files:
            continue
        indexes = sorted({len(files) // 3, len(files) * 2 // 3})
        images = []
        for index in indexes:
            image = cv2.imread(str(files[index]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            images.append(cv2.resize(image, (300, tile_height), interpolation=cv2.INTER_AREA))
        if not images:
            continue
        if len(images) == 1:
            images.append(images[0].copy())
        tile = np.hstack(images[:2])
        cv2.rectangle(tile, (0, 0), (tile_width, 28), (255, 255, 255), -1)
        cv2.putText(
            tile,
            f'{entry["rarity"]} {entry["slug"]}',
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        print('[overview] no extracted frames')
        return 1
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        x = index % columns * tile_width
        y = index // columns * tile_height
        sheet[y:y + tile_height, x:x + tile_width] = tile
    destination = root / 'reference_midframes.jpg'
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'[overview] {len(tiles)} classes: {destination}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare new Hyakkiyakou shikigami materials.')
    parser.add_argument('action', choices=('download', 'extract', 'all', 'status', 'overview'))
    parser.add_argument('--name', action='append', default=[], help='Chinese name or manifest slug; repeatable')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--cookies-from-browser', help='Optional yt-dlp browser name, for example chrome')
    parser.add_argument('--interval', type=float, default=0.5, help='Seconds between candidate frames')
    parser.add_argument('--dedupe-threshold', type=float, default=2.0, help='Minimum frame signature difference')
    args = parser.parse_args()
    entries = load_entries(args.name)
    root = args.root.resolve()

    failures = 0
    if args.action in ('download', 'all'):
        failures = download(entries, root, args.cookies_from_browser)
    if args.action in ('extract', 'all'):
        extract(entries, root, args.interval, args.dedupe_threshold)
    if args.action == 'status' or args.action == 'all':
        status(entries, root)
    if args.action == 'overview':
        return overview(entries, root)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
