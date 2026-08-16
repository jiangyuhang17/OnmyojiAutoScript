import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from oashya.labels import CLASSINDEX as CI, id2label, id2name
from oashya.tracker import Tracker


DEFAULT_SOURCE = Path('log/hya')
DEFAULT_OUTPUT = Path('tasks/Hyakkiyakou/dataset/new_shikigami/real_hyakki')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def class_info(class_id: int) -> tuple[str, str]:
    try:
        return id2label(class_id), id2name(class_id)
    except (IndexError, KeyError):
        return f'class_{class_id}', f'class_{class_id}'


def crop_track(image: np.ndarray, track: tuple, padding: float) -> np.ndarray | None:
    _, _, _, cx, cy, width, height, _ = track
    half_width = width * (0.5 + padding)
    half_height = height * (0.5 + padding)
    x1 = max(0, int(cx - half_width))
    y1 = max(0, int(cy - half_height))
    x2 = min(image.shape[1], int(cx + half_width))
    y2 = min(image.shape[0], int(cy + half_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def load_processed(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    processed = set()
    for line in index_path.read_text(encoding='utf-8').splitlines():
        try:
            processed.add(json.loads(line)['source'])
        except (json.JSONDecodeError, KeyError):
            continue
    return processed


def extract(source: Path, output: Path, confidence: float, padding: float) -> None:
    image_paths = sorted(
        path for path in source.rglob('*') if path.suffix.lower() in IMAGE_SUFFIXES
    )
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / 'candidates.jsonl'
    processed = load_processed(index_path)
    tracker = Tracker(args={
        'conf_threshold': confidence,
        'iou_threshold': 0.7,
        'precision': 'fp32',
        'inference_engine': 'onnxruntime',
        'debug': False,
    })

    written = 0
    skipped = 0
    with index_path.open('a', encoding='utf-8') as index_file:
        for image_path in image_paths:
            source_key = image_path.resolve().as_posix()
            if source_key in processed:
                skipped += 1
                continue
            image = read_image(image_path)
            if image is None:
                continue
            tracks = tracker(image=image, response=[0, 0, False, 10])
            record = {'source': source_key, 'tracks': []}
            for track_index, track in enumerate(tracks):
                track_id, class_id, conf, cx, cy, width, height, velocity = track
                label, name = class_info(class_id)
                item = {
                    'track_id': track_id,
                    'predicted_class': class_id,
                    'predicted_label': label,
                    'predicted_name': name,
                    'confidence': round(float(conf), 6),
                    'xywh': [int(cx), int(cy), int(width), int(height)],
                    'velocity': round(float(velocity), 6),
                }
                if class_id > CI.MAX_BUFF:
                    crop = crop_track(image, track, padding)
                    if crop is not None:
                        crop_dir = output / 'candidates' / label
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_name = f'{image_path.stem}_{track_index:02d}.jpg'
                        crop_path = crop_dir / crop_name
                        encoded_image = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        cv2.imencode('.jpg', encoded_image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(crop_path)
                        item['crop'] = crop_path.resolve().as_posix()
                        written += 1
                record['tracks'].append(item)
            index_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            tracker.clear_tracks()

    print(f'[candidates] {len(image_paths)} images, {written} crops, {skipped} skipped')
    print(f'[index] {index_path}')


def status(source: Path, output: Path) -> None:
    images = sum(1 for path in source.rglob('*') if path.suffix.lower() in IMAGE_SUFFIXES) if source.exists() else 0
    crops = sum(1 for path in (output / 'candidates').rglob('*.jpg')) if (output / 'candidates').exists() else 0
    indexed = len(load_processed(output / 'candidates.jsonl'))
    print(f'[status] source images={images}, indexed={indexed}, candidate crops={crops}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect candidate crops from real Hyakkiyakou screenshots.')
    parser.add_argument('action', choices=('extract', 'status'))
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--padding', type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == 'extract':
        extract(args.source, args.output, args.confidence, args.padding)
    else:
        status(args.source, args.output)


if __name__ == '__main__':
    main()
