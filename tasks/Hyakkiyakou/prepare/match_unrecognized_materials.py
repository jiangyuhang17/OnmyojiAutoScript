"""Rank unrecognized real captures against official new-shikigami materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "tasks" / "Hyakkiyakou" / "dataset" / "new_shikigami"


def image_batches(paths: list[Path], preprocess, batch_size: int):
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        yield batch_paths, torch.stack(images)


def embed(paths: list[Path], model, preprocess, device: str, batch_size: int) -> torch.Tensor:
    features = []
    with torch.inference_mode():
        for _, images in image_batches(paths, preprocess, batch_size):
            values = model.encode_image(images.to(device))
            features.append(values / values.norm(dim=-1, keepdim=True))
    return torch.cat(features).cpu()


def official_prototypes(
    frame_root: Path,
    model,
    preprocess,
    device: str,
    sample_count: int,
    batch_size: int,
) -> tuple[list[str], torch.Tensor]:
    names = []
    prototypes = []
    for rarity in ("sp", "ssr"):
        rarity_dir = frame_root / rarity
        if not rarity_dir.exists():
            continue
        for class_dir in sorted(path for path in rarity_dir.iterdir() if path.is_dir()):
            paths = sorted(class_dir.rglob("*.jpg"))
            if not paths:
                continue
            indexes = np.linspace(0, len(paths) - 1, min(sample_count, len(paths)), dtype=int)
            sampled = [paths[index] for index in indexes]
            features = embed(sampled, model, preprocess, device, batch_size)
            prototype = features.mean(dim=0)
            prototypes.append(prototype / prototype.norm())
            names.append(class_dir.name)
    return names, torch.stack(prototypes)


def write_sheets(matches: dict[str, list[dict]], candidate_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 300, 230
    for name, items in matches.items():
        canvas = np.full((2 * tile_height, 5 * tile_width, 3), 245, dtype=np.uint8)
        for index, item in enumerate(items[:10]):
            image = cv2.imread(str(candidate_dir / item["crop"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            scale = min((tile_width - 8) / image.shape[1], (tile_height - 32) / image.shape[0])
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            row, column = divmod(index, 5)
            x = column * tile_width + (tile_width - resized.shape[1]) // 2
            y = row * tile_height + 26 + (tile_height - 26 - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            label = f"{item['score']:.3f} {item['candidate_id']:04d}"
            cv2.putText(canvas, label, (column * tile_width + 5, row * tile_height + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Directory containing index.json and candidate crops; defaults to real_hyakki/unrecognized",
    )
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit("Install analysis dependency: pip install open-clip-torch") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.eval().to(device)

    real_dir = args.candidate_dir or args.dataset / "real_hyakki" / "unrecognized"
    index = json.loads((real_dir / "index.json").read_text(encoding="ascii"))
    candidate_paths = [real_dir / item["crop"] for item in index]
    candidate_features = embed(candidate_paths, model, preprocess, device, args.batch_size)
    names, prototypes = official_prototypes(
        args.dataset / "frames", model, preprocess, device, args.sample_count, args.batch_size
    )
    similarities = prototypes @ candidate_features.T

    matches = {}
    for class_index, name in enumerate(names):
        values, indexes = similarities[class_index].topk(min(10, len(candidate_paths)))
        matches[name] = [
            {
                "candidate_id": int(candidate_index),
                "score": round(float(score), 6),
                "source": index[int(candidate_index)]["source"],
                "crop": index[int(candidate_index)]["crop"],
            }
            for score, candidate_index in zip(values, indexes)
        ]

    output_dir = real_dir / "material_matches"
    write_sheets(matches, real_dir, output_dir)
    (output_dir / "matches.json").write_text(json.dumps(matches, indent=2) + "\n", encoding="ascii")
    print(f"Matched {len(candidate_paths)} candidates against {len(names)} classes on {device}")


if __name__ == "__main__":
    main()
