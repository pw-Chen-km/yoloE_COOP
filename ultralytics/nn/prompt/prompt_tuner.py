# trainers/coop_trainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ultralytics.nn.text_model import build_text_model  # 你的 wrapper
from ultralytics.nn.prompt.prompt_learner import PromptLearner           # 你的 PromptLearner

class CoOpTrainer:
    def __init__(self, classnames, variant="clip:ViT-B/32", device="cuda",
                 lr=2e-3, wd=0.0, fp32=True,
                 num_ctx: int = 16,
                 ctx_init: str | None = None,
                 per_class: bool = False,
                 class_token_position: str = "end",
                 trainable_ctx: bool = True,
                 add_period: bool = True,
                 optimize_to_text: bool = False,
                 ):
        self.device = torch.device(device)
        self.text = build_text_model(variant, device=self.device)      # CLIP/MobileCLIP wrapper
        self.classnames = classnames

        # 建立 PromptLearner：直接傳 wrapper（統一暴露 token_embedding/positional_embedding 等）
        self.pl = PromptLearner(
            classnames,
            self.text,
            num_ctx=num_ctx,
            ctx_init=ctx_init,
            per_class=per_class,
            class_token_position=class_token_position,
            dtype=None,
            trainable_ctx=trainable_ctx,
            add_period=add_period,
        ).to(self.device)

        # 凍結文本/影像編碼器
        for p in self.text.parameters():
            p.requires_grad_(False)

        self.optim = torch.optim.AdamW(self.pl.parameters(), lr=lr, weight_decay=wd)
        self.fp32 = fp32
        self.optimize_to_text = bool(optimize_to_text)

        # 若啟用 optimize_to_text，預先計算原始文本嵌入作為目標
        self.base_txt = None
        if self.optimize_to_text:
            names_for_tok = [
                (name + ("." if add_period else "")) for name in self.classnames
            ]
            with torch.no_grad():
                tok = self.text.tokenize(names_for_tok)
                base = self.text.encode_text(tok, dtype=torch.float32)  # [C, D]
                base = F.normalize(base, dim=-1)
            self.base_txt = base.to(self.device)

    def contrastive_loss(self, img_feats, txt_feats, logit_scale=None):
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        scale = (self.text.logit_scale.exp().item()
                 if hasattr(self.text, "logit_scale") and self.text.logit_scale is not None
                 else 100.0)  # CLIP 典型 scale
        if logit_scale is not None:
            scale = logit_scale
        logits = scale * img_feats @ txt_feats.t()   # [B, C]
        targets = torch.arange(img_feats.size(0), device=img_feats.device) % txt_feats.size(0)
        return F.cross_entropy(logits, targets)

    def step(self, images, labels):
        # 1) image feats（不回傳梯度）
        with torch.no_grad():
            img_feats = self.text.encode_image(images, dtype=torch.float32 if self.fp32 else images.dtype)

        # 2) build prompts & text feats（要回傳梯度）
        prompts = self.pl()                                            # [C, T, d]
        token_ids = self.pl.tokenized_prompts.to(self.device)          # [C, T]
        txt_feats = self.text.encode_from_embeddings(
            prompts, token_ids, add_positional=True, dtype=torch.float32
        )                                                               # [C, D]

        # 3) 損失計算
        #print(f"[debug] txt_feats.shape={tuple(txt_feats.shape)}")
        #print(f"[debug] img_feats.shape={tuple(img_feats.shape)}")
        #print(f"[debug] labels={labels}")
        cls_txt = txt_feats[labels]                                     # [B, D]
        if self.optimize_to_text:
            assert self.base_txt is not None
            cls_target = self.base_txt[labels]                          # [B, D]
            loss = (1.0 - F.cosine_similarity(cls_txt, cls_target, dim=-1)).mean()
        else:
            # 當僅有單一類別時，改用相似度最大化損失，避免無負樣本導致退化
            if getattr(self.pl, "n_cls", None) == 1:
                cos = F.cosine_similarity(img_feats, cls_txt, dim=-1)
                loss = 100*(1.0 - cos).mean()
            else:
                loss = self.contrastive_loss(img_feats, cls_txt)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def _crop_roi(self, image, bbox=None):
        """以 bbox 裁剪 ROI；若無 bbox 則回傳原圖。"""
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image")
        if bbox is None:
            return image
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = image.size
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return image.crop((x1, y1, x2, y2))

    def tune(self, image, bbox=None, steps=100, target_index=0, save_roi_path=None):
        """單圖 ROI prompt tuning，多步更新 learnable prompts，返回 [1, C, D] 的文字原型。"""
        preprocess = getattr(self.text, "preprocess", None)
        if preprocess is None:
            raise RuntimeError("Text model does not expose preprocess; cannot prepare ROI.")

        roi = self._crop_roi(image, bbox)
        if save_roi_path is not None:
            try:
                roi.save(save_roi_path)
                print(f"[tune] ROI saved to: {save_roi_path}")
            except Exception as e:
                print(f"[tune] Failed to save ROI to {save_roi_path}: {e}")
        images = preprocess(roi).unsqueeze(0).to(self.device)
        labels = torch.tensor([int(target_index)], device=self.device, dtype=torch.long)

        total_loss = 0.0
        steps = int(steps)
        for i in range(steps):
            loss = self.step(images, labels)
            total_loss += float(loss)
            # 進度列印（每 10 步一次）
            if (i + 1) % 10 == 0 or (i + 1) == steps:
                print(f"[tune] step {i + 1}/{steps} loss={loss:.4f}")

        avg_loss = total_loss / max(steps, 1)
        print(f"[tune] avg_loss={avg_loss:.4f}")

        return self.export_tpe()  # [1, C, D]

    def fit(self, dataloader: DataLoader, epochs: int = 1):
        """多批次資料訓練 learnable prompts；結束後返回 [1, C, D]。"""
        epochs = int(epochs)
        for ep in range(epochs):
            total_loss = 0.0
            num = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                loss = self.step(images, labels)
                total_loss += float(loss)
                num += 1
            avg = total_loss / max(num, 1)
            print(f"[fit] epoch {ep + 1}/{epochs} avg_loss={avg:.4f}")
        return self.export_tpe()

    @torch.no_grad()
    def export_tpe(self):
        """輸出 [1, C, D] 的文字原型（後續你要給 YOLOE 用也很方便）。"""
        prompts = self.pl()
        token_ids = self.pl.tokenized_prompts.to(self.device)
        txt = self.text.encode_from_embeddings(prompts, token_ids, add_positional=True, dtype=torch.float32)
        txt = F.normalize(txt, dim=-1)            # [C, D]
        return txt.unsqueeze(0)                    # [1, C, D]