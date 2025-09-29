import argparse
import os

import torch
from PIL import Image
import supervision as sv

from ultralytics import YOLOE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yoloe-v8l-seg.pt",
        help="Path or ID of the model checkpoint",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["person"],
        help="List of class names to set for the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--prompt-image",
        type=str,
        help="Optional visual prompt image used to tune text embeddings online",
    )
    parser.add_argument(
        "--prompt-boxes",
        type=float,
        nargs="+",
        help="Bounding boxes for the prompt image (x1 y1 x2 y2 per class)",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="xyxy",
        choices=["xyxy", "xywh"],
        help="Format of the provided prompt boxes",
    )
    parser.add_argument(
        "--prompt-normalized",
        action="store_true",
        help="Indicate that prompt boxes are already normalized to [0, 1]",
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=None,
        help="Override the support crop resolution used for visual prompt tuning",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.source)
        args.output = f"{base}-output{ext}"

    image = Image.open(args.source).convert("RGB")

    model = YOLOE(args.checkpoint)
    model.to(args.device)

    if args.prompt_image:
        if not args.prompt_boxes:
            raise ValueError("--prompt-boxes must be provided when using --prompt-image.")
        prompt_image = Image.open(args.prompt_image).convert("RGB")
        box_values = torch.tensor(args.prompt_boxes, dtype=torch.float32).view(-1, 4)
        if box_values.shape[0] != len(args.names):
            raise ValueError("Number of prompt boxes must match the number of class names.")
        model.tune_text_prompt(
            visual_prompt=prompt_image,
            boxes=box_values,
            class_names=args.names,
            box_format=args.prompt_format,
            normalized=args.prompt_normalized,
            support_size=args.support_size,
        )
    else:
        model.set_classes(args.names, model.get_text_pe(args.names))

    results = model.predict(image, verbose=False)

    detections = sv.Detections.from_ultralytics(results[0])

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4,
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness,
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True,
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    annotated_image.save(args.output)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
