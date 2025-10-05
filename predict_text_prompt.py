import argparse
import os
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from ultralytics.nn.prompt.prompt_tuner import CoOpTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yoloe-v8l-seg.pt",
        help="Path or ID of the model checkpoint"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["person"],
        help="List of class names to set for the model"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        help="ROI bbox as 4 integers: x1 y1 x2 y2"
    )
    parser.add_argument(
        "--tune-steps",
        type=int,
        default=100,
        help="Number of tuning steps for prompt learning"
    )
    parser.add_argument(
        "--tune-lr",
        type=float,
        default=2e-3,
        help="Learning rate for prompt tuning"
    )
    parser.add_argument(
        "--tune-target",
        type=int,
        default=0,
        help="Target class index during single-image tuning"
    )
    parser.add_argument(
        "--text-model",
        type=str,
        help="Text model variant, e.g. clip:ViT-B/32 or mobileclip:blt"
    )
    parser.add_argument(
        "--ctx-init",
        type=str,
        help="Prompt context init string, e.g. 'a photo of a'"
    )
    parser.add_argument(
        "--ctx-trainable",
        type=int,
        default=1,
        help="Whether context is trainable (1) or frozen (0)"
    )
    parser.add_argument(
        "--no-period",
        action="store_true",
        help="Remove trailing period after class name in COOP prompts"
    )
    parser.add_argument(
        "--tune-to-text",
        action="store_true",
        help="Optimize learnable prompts to match raw text embeddings"
    )
    parser.add_argument(
        "--save-roi",
        type=str,
        help="Path to save cropped ROI image (effective when tuning with --bbox)"
    )
    parser.add_argument(
        "--debug-text-as-learnable",
        action="store_true",
        help="Use text encoder PE as if it were learnable PE (debug pipeline)")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on"
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

    def get_learnable_prompt_pe(image, names, model, steps=100, target_index=0, variant=None, bbox=None, lr=2e-3, save_roi_path=None):
        # 決定文字模型 variant 與裝置
        device = next(model.parameters()).device if hasattr(model, "parameters") else None
        device_str = device.type if device is not None else args.device
        if variant is None:
            variant = getattr(model, "args", {}).get("text_model", None)
        if variant is None:
            variant = "clip:ViT-B/32"

        # DEBUG: 以 text encoder 的輸出假裝 learnable 的 raw，再走與 learnable 相同的 head.get_tpe
        if args.debug_text_as_learnable:
            from ultralytics.nn.text_model import build_text_model
            text_model = build_text_model(variant, device=device_str)
            text_token = text_model.tokenize(names)
            txt = text_model.encode_text(text_token)  # [C, D]
            raw = txt.reshape(1, len(names), txt.shape[-1]).clone()
            inner = getattr(model, "model", None)
            head_seq = getattr(inner, "model", inner)
            head = head_seq[-1]
            return head.get_tpe(raw)

        # 單圖 ROI 調參，取回 raw [1, C, D]
        trainer = CoOpTrainer(
            names,
            variant=variant,
            device=device_str,
            lr=lr,
            num_ctx=16,
            ctx_init=args.ctx_init,
            trainable_ctx=bool(args.ctx_trainable),
            add_period=not args.no_period,
            optimize_to_text=bool(args.tune_to_text),
            # 新增 YOLOE 對齊參數
            yoloe_model=model,
            det_preprocess=getattr(model, "preprocess", None),
            logit_scale=100.0,
            mil_reduce="max",
        )
        raw = trainer.tune(
            image,
            bbox=bbox,
            steps=int(steps),
            target_index=int(target_index),
            save_roi_path=save_roi_path,
        )

        # 對齊至檢測頭空間
        # 兼容兩種結構：
        # 1) model.model 是完整 YOLOE*Model（其內部還有 .model 為 nn.Sequential）
        # 2) model.model 已經是 nn.Sequential
        inner = getattr(model, "model", None)
        head_seq = getattr(inner, "model", inner)
        head = head_seq[-1]
        tuned = head.get_tpe(raw)
        return tuned

    model.set_classes(
        args.names,
        get_learnable_prompt_pe(
            image,
            args.names,
            model,
            steps=args.tune_steps,
            target_index=args.tune_target,
            variant=args.text_model,
            bbox=args.bbox,
            lr=args.tune_lr,
            save_roi_path=args.save_roi,
        ),
    )
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
        opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    annotated_image.save(args.output)
    print(f"Annotated image saved to: {args.output}")

if __name__ == "__main__":
    main()
