"""Example script for offline CoOp prompt tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image

from ultralytics.nn.prompt_tuning import COOPPromptTuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoOp prompt tuning on a small support set.")
    parser.add_argument("--support", type=Path, default=None, help="Directory with class sub-folders of support images.")
    parser.add_argument("--class-names", type=str, nargs="*", default=None, help="Explicit class names to tune.")
    parser.add_argument("--text-model", type=str, default="clip:ViT-B/32", help="Base text model variant.")
    parser.add_argument("--ctx-len", type=int, default=16, help="Number of learnable context tokens.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for context optimization.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--output", type=Path, default=Path("runs/coop/prompt_state.pt"), help="Path to save tuner state.")
    parser.add_argument("--load-state", type=Path, default=None, help="Optional tuner state to load before tuning.")
    return parser.parse_args()


def discover_class_names(support_dir: Path) -> List[str]:
    return sorted([p.name for p in support_dir.iterdir() if p.is_dir()])


def load_support_images(
    support_dir: Path, class_names: Sequence[str], tuner: COOPPromptTuner
) -> List[tuple[torch.Tensor, int]]:
    preprocess = getattr(tuner.base_model, "preprocess", None)
    if preprocess is None:
        raise RuntimeError("Base model does not expose a preprocess transform for loading support images.")

    support: List[tuple[torch.Tensor, int]] = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_index, name in enumerate(class_names):
        image_dir = support_dir / name
        if not image_dir.exists():
            raise FileNotFoundError(f"Support directory missing class folder: {image_dir}")
        for image_path in sorted(image_dir.glob("**/*")):
            if image_path.is_file() and image_path.suffix.lower() in valid_ext:
                image = Image.open(image_path).convert("RGB")
                tensor = preprocess(image).unsqueeze(0)
                support.append((tensor, class_index))
    if not support:
        raise RuntimeError(f"No support images found under {support_dir}.")
    return support


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tuner = COOPPromptTuner(args.text_model, device=device, ctx_len=args.ctx_len)

    if args.load_state is not None:
        tuner.load_state(args.load_state, map_location=device)

    class_names = args.class_names
    if class_names is None and args.support is not None:
        class_names = discover_class_names(args.support)
    if class_names is None:
        class_names = tuner.class_names or []

    if args.support is not None and class_names:
        support = load_support_images(args.support, class_names, tuner)
        tuner.fit(support, class_names, steps=args.steps, lr=args.lr)

    if class_names:
        print("Tuned prompts for classes:", ", ".join(class_names))
        features = tuner.get_text_features()
        print("Feature tensor shape:", tuple(features.shape))

    if args.output is not None and tuner.context is not None:
        tuner.save_state(args.output)


if __name__ == "__main__":
    main()
