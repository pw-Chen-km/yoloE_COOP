"""Visual prompt tuning GUI for YOLOE COOP models.

This script provides a minimal Tkinter-based GUI that allows a user to:

* Select a pretrained YOLOE checkpoint and load it on the available device.
* Open a visual prompt image and draw bounding boxes describing objects of
  interest.
* Provide optional class names and support size overrides for COOP tuning.
* Open a target image and run prediction using the tuned text prompts.
* Preview the annotated detection+segmentation results and save them.

The implementation mirrors the snippet shared in the issue discussion so that
users can quickly validate the regression fix around missing ``coop_tuner``
attributes when loading legacy checkpoints.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import torch
import supervision as sv
from PIL import Image, ImageTk

from ultralytics import YOLOE


class VisualPromptGUI(tk.Tk):
    """Minimal GUI to tune COOP prompts from a visual support image."""

    def __init__(self) -> None:
        super().__init__()
        self.title("YOLOE COOP Visual Prompt GUI")
        self.geometry("1400x820")

        # Model state
        self.model: YOLOE | None = None
        default_ckpt = "pretrain/yoloe-v8l-seg.pt"
        self.checkpoint_path = default_ckpt if os.path.exists(default_ckpt) else "yoloe-v8l-seg.pt"
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Prompt image state
        self.prompt_image_path: str | None = None
        self.prompt_image_pil: Image.Image | None = None
        self.prompt_display_imgtk: ImageTk.PhotoImage | None = None
        self.prompt_scale: float = 1.0

        # Target image state
        self.target_image_path: str | None = None
        self.target_image_pil: Image.Image | None = None
        self.target_display_imgtk: ImageTk.PhotoImage | None = None
        self.target_scale: float = 1.0

        # Drawing state
        self.box_start_xy: tuple[int, int] | None = None
        self.temp_rect_id: int | None = None
        self.prompt_boxes_xyxy: list[tuple[float, float, float, float]] = []

        # Result state
        self.result_image_pil: Image.Image | None = None
        self.result_display_imgtk: ImageTk.PhotoImage | None = None

        self._build_widgets()

    # UI construction ------------------------------------------------------------------------------
    def _build_widgets(self) -> None:
        # Top controls
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Label(top, text="Checkpoint:").pack(side=tk.LEFT)
        self.ckpt_var = tk.StringVar(value=self.checkpoint_path)
        self.ckpt_entry = tk.Entry(top, textvariable=self.ckpt_var, width=60)
        self.ckpt_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Browse", command=self._on_browse_ckpt).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Load Model", command=self._on_load_model).pack(side=tk.LEFT, padx=10)

        device_label = (
            f"Device: {self.device_str}"
            f" ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'})"
        )
        self.device_var = tk.StringVar(value=device_label)
        tk.Label(top, textvariable=self.device_var, fg="#2a6").pack(side=tk.LEFT, padx=15)

        # Middle split: prompt on the left, target/result on the right
        middle = tk.Frame(self)
        middle.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        left = tk.LabelFrame(middle, text="Visual Prompt")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        prompt_toolbar = tk.Frame(left)
        prompt_toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(prompt_toolbar, text="Open Prompt Image", command=self._on_open_prompt_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        tk.Button(prompt_toolbar, text="Clear Boxes", command=self._on_clear_boxes).pack(side=tk.LEFT, padx=5)
        tk.Label(prompt_toolbar, text="Draw boxes with mouse: Press-Drag-Release").pack(side=tk.LEFT, padx=10)

        self.prompt_canvas = tk.Canvas(left, bg="#222", width=640, height=480, cursor="cross")
        self.prompt_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.prompt_canvas.bind("<ButtonPress-1>", self._on_prompt_mouse_down)
        self.prompt_canvas.bind("<B1-Motion>", self._on_prompt_mouse_move)
        self.prompt_canvas.bind("<ButtonRelease-1>", self._on_prompt_mouse_up)

        prompt_bottom = tk.Frame(left)
        prompt_bottom.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(prompt_bottom, text="Class names (comma-separated):").grid(row=0, column=0, sticky="w")
        self.names_var = tk.StringVar(value="person,glasses")
        tk.Entry(prompt_bottom, textvariable=self.names_var, width=40).grid(row=0, column=1, sticky="w", padx=5)

        tk.Label(prompt_bottom, text="Support size (optional):").grid(row=1, column=0, sticky="w")
        self.support_size_var = tk.StringVar(value="")
        tk.Entry(prompt_bottom, textvariable=self.support_size_var, width=12).grid(row=1, column=1, sticky="w")

        self.boxes_info_var = tk.StringVar(value="Boxes: []")
        tk.Label(prompt_bottom, textvariable=self.boxes_info_var, fg="#06c").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        # Right side: target and result canvases
        right = tk.LabelFrame(middle, text="Target & Result")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tgt_toolbar = tk.Frame(right)
        tgt_toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(tgt_toolbar, text="Open Target Image", command=self._on_open_target_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        tk.Button(tgt_toolbar, text="Run Inference", command=self._on_run_inference).pack(side=tk.LEFT, padx=10)
        tk.Button(tgt_toolbar, text="Save Result", command=self._on_save_result).pack(side=tk.LEFT, padx=5)

        self.target_canvas = tk.Canvas(right, bg="#222", width=640, height=320)
        self.target_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_canvas = tk.Canvas(right, bg="#111", width=640, height=320)
        self.result_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Event handlers ---------------------------------------------------------------------------------
    def _on_browse_ckpt(self) -> None:
        initial_dir = os.path.abspath("pretrain") if os.path.exists("pretrain") else os.getcwd()
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select checkpoint",
            filetypes=[
                ("PyTorch weights", "*.pt;*.pth;*.ptl;*.torchscript"),
                ("All", "*.*"),
            ],
        )
        if path:
            self.ckpt_var.set(path)

    def _on_load_model(self) -> None:
        ckpt = self.ckpt_var.get().strip()
        if not ckpt:
            messagebox.showerror("Error", "Please select a checkpoint file.")
            return
        try:
            self.model = YOLOE(ckpt)
            self.model.to(self.device_str)
            messagebox.showinfo("Model", f"Model loaded on {self.device_str}.")
        except Exception as exc:  # noqa: BLE001
            self.model = None
            messagebox.showerror("Load failed", f"Failed to load model:\n{exc}")

    def _on_open_prompt_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select visual prompt image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to open image:\n{exc}")
            return

        self.prompt_image_path = path
        self.prompt_image_pil = img
        self.prompt_boxes_xyxy.clear()
        self._update_boxes_info()
        self._draw_on_canvas(image=img, canvas=self.prompt_canvas, is_prompt=True)

    def _on_open_target_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select target image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to open image:\n{exc}")
            return

        self.target_image_path = path
        self.target_image_pil = img
        self._draw_on_canvas(image=img, canvas=self.target_canvas, is_prompt=False)

    def _on_save_result(self) -> None:
        if self.result_image_pil is None:
            messagebox.showwarning("No result", "Please run inference first.")
            return
        initial_dir = os.path.dirname(self.target_image_path) if self.target_image_path else os.getcwd()
        out_path = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if out_path:
            try:
                self.result_image_pil.save(out_path)
                messagebox.showinfo("Saved", f"Saved to: {out_path}")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", f"Failed to save image:\n{exc}")

    def _on_clear_boxes(self) -> None:
        self.prompt_boxes_xyxy.clear()
        self._update_boxes_info()
        self._redraw_prompt_canvas()

    def _on_prompt_mouse_down(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.prompt_image_pil is None:
            return
        self.box_start_xy = (event.x, event.y)
        if self.temp_rect_id:
            self.prompt_canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

    def _on_prompt_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.prompt_image_pil is None or self.box_start_xy is None:
            return
        x0, y0 = self.box_start_xy
        x1, y1 = event.x, event.y
        if self.temp_rect_id:
            self.prompt_canvas.coords(self.temp_rect_id, x0, y0, x1, y1)
        else:
            self.temp_rect_id = self.prompt_canvas.create_rectangle(
                x0, y0, x1, y1, outline="#00ff88", width=2
            )

    def _on_prompt_mouse_up(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.prompt_image_pil is None or self.box_start_xy is None:
            return
        x0, y0 = self.box_start_xy
        x1, y1 = event.x, event.y
        self.box_start_xy = None

        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        if x_max - x_min < 3 or y_max - y_min < 3:
            if self.temp_rect_id:
                self.prompt_canvas.delete(self.temp_rect_id)
                self.temp_rect_id = None
            return

        scale = self.prompt_scale
        x_min_o = x_min / scale
        y_min_o = y_min / scale
        x_max_o = x_max / scale
        y_max_o = y_max / scale
        self.prompt_boxes_xyxy.append((x_min_o, y_min_o, x_max_o, y_max_o))

        if self.temp_rect_id is not None:
            self.prompt_canvas.itemconfig(self.temp_rect_id, outline="#00ff00")
            self.temp_rect_id = None
        self._update_boxes_info()

    # Core logic -------------------------------------------------------------------------------------
    def _draw_on_canvas(self, image: Image.Image, canvas: tk.Canvas, is_prompt: bool) -> None:
        canvas_width = int(canvas.winfo_width() or canvas["width"])  # type: ignore[index]
        canvas_height = int(canvas.winfo_height() or canvas["height"])  # type: ignore[index]
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 640, 480

        img_w, img_h = image.size
        scale = min(canvas_width / img_w, canvas_height / img_h)
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))
        resized = image.resize((disp_w, disp_h), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        if is_prompt:
            self.prompt_display_imgtk = imgtk
            self.prompt_scale = scale
            for x0, y0, x1, y1 in self.prompt_boxes_xyxy:
                canvas.create_rectangle(x0 * scale, y0 * scale, x1 * scale, y1 * scale, outline="#00ff00", width=2)
        else:
            self.target_display_imgtk = imgtk
            self.target_scale = scale

    def _redraw_prompt_canvas(self) -> None:
        if self.prompt_image_pil is None:
            return
        self._draw_on_canvas(image=self.prompt_image_pil, canvas=self.prompt_canvas, is_prompt=True)

    def _update_boxes_info(self) -> None:
        if not self.prompt_boxes_xyxy:
            self.boxes_info_var.set("Boxes: []")
        else:
            parts = [f"[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]" for x0, y0, x1, y1 in self.prompt_boxes_xyxy]
            self.boxes_info_var.set("Boxes: " + ", ".join(parts))

    def _get_class_names(self) -> list[str]:
        raw = self.names_var.get().strip()
        if not raw:
            return []
        names = [segment.strip() for segment in raw.replace("\n", ",").replace("\t", ",").split(",")]
        return [name for name in names if name]

    def _on_run_inference(self) -> None:
        if self.model is None:
            messagebox.showwarning("Model", "Load the model first.")
            return
        if self.prompt_image_pil is None or not self.prompt_boxes_xyxy:
            messagebox.showwarning("Prompt", "Open a prompt image and draw at least one box.")
            return
        if self.target_image_pil is None:
            messagebox.showwarning("Target", "Open a target image.")
            return

        class_names = self._get_class_names()
        total_boxes = len(self.prompt_boxes_xyxy)
        if total_boxes == 0:
            messagebox.showwarning("Prompt", "Draw at least one box.")
            return

        if not class_names:
            class_names = [f"object{i}" for i in range(total_boxes)]
        elif len(class_names) < total_boxes:
            class_names.extend(f"object{i}" for i in range(len(class_names), total_boxes))
        elif len(class_names) > total_boxes:
            class_names = class_names[:total_boxes]
        self.names_var.set(",".join(class_names))

        selected_boxes = np.array(self.prompt_boxes_xyxy[: len(class_names)], dtype=np.float32)
        boxes_tensor = torch.tensor(selected_boxes, dtype=torch.float32)

        try:
            support_size_val = int(self.support_size_var.get()) if self.support_size_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Support size", "Support size must be an integer.")
            return

        try:
            self.model.tune_text_prompt(
                visual_prompt=self.prompt_image_pil,
                boxes=boxes_tensor,
                class_names=class_names,
                box_format="xyxy",
                normalized=False,
                support_size=support_size_val,
                tuner_cfg={"base_variant": "clip:ViT-B/32"},
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Tune failed", f"Failed to tune text prompt:\n{exc}")
            return

        try:
            results = self.model.predict(self.target_image_pil, verbose=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Predict failed", f"Failed to run prediction:\n{exc}")
            return

        try:
            detections = sv.Detections.from_ultralytics(results[0])

            labels = [
                f"{cls} {conf:.2f}"
                for cls, conf in zip(detections["class_name"], detections.confidence)
            ]

            annotated = self.target_image_pil.copy()
            annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(
                scene=annotated, detections=detections
            )
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=self.target_image_pil.size)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=self.target_image_pil.size)
            annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(
                scene=annotated, detections=detections
            )
            annotated = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True
            ).annotate(scene=annotated, detections=detections, labels=labels)

            self.result_image_pil = annotated
            self._draw_on_canvas(image=annotated, canvas=self.result_canvas, is_prompt=False)
            messagebox.showinfo("Done", "Inference completed.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Annotate failed", f"Failed to annotate results:\n{exc}")


def main() -> None:
    app = VisualPromptGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
