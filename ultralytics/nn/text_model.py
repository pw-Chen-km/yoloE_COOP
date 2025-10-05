from abc import abstractmethod
import clip
import mobileclip
import torch.nn as nn
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from ultralytics.utils import LOGGER

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()

    # text API
    @abstractmethod
    def tokenize(self, texts):
        pass

    @abstractmethod
    def encode_text(self, token_ids, dtype):
        pass

    # CoOp: run text encoder from pre-built embeddings (keep grads!)
    def encode_from_embeddings(self, emb_seq, token_ids, **kwargs):
        raise NotImplementedError

    # image API
    def encode_image(self, images, dtype):
        raise NotImplementedError

    # optional: expose preprocess for dataloaders
    @property
    def preprocess(self):
        return getattr(self, "_preprocess", None)

    # optional: expose logit scale (CLIP has one)
    @property
    def logit_scale(self):
        return None


class CLIP(TextModel):
    """
    CLIP 文字模型包裝：
    - tokenize(): 用 OpenAI CLIP 的 tokenizer
    - encode_text(): 原生文字編碼（不做 prompt learning）
    - encode_from_embeddings(): 從已組好的 embedding 序列跑完整個文字編碼器（CoOp 用）
    """
    def __init__(self, size, device):
        super().__init__()
        model, preprocess = clip.load(size, device=device, jit=False)
        self.model = model.eval()
        self.device = device
        self.to(device)
        # expose preprocess
        self._preprocess = preprocess

        # === 對 PromptLearner 暴露關鍵屬性 ===
        self.token_embedding = self.model.token_embedding            # nn.Embedding
        self.positional_embedding = self.model.positional_embedding  # [context_length, d]
        self.ln_final = self.model.ln_final
        self.text_projection = self.model.text_projection            # [d, proj]
        self.context_length = self.model.context_length              # int
        self.dtype = self.model.dtype                                # torch.float16/32 (依權重而定)

    def tokenize(self, texts):
        return clip.tokenize(texts).to(self.device)

    @smart_inference_mode()  # 推論用：不回傳梯度
    def encode_text(self, token_ids, dtype=torch.float32):
        """
        OpenAI CLIP 的原生 encode_text（給「非 prompt learning」路徑）
        token_ids: [N, T]
        """
        x = self.model.encode_text(token_ids).to(dtype)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x

    # 注意：不要加 smart_inference_mode()，需讓梯度能回傳到 learnable prompts
    def encode_from_embeddings(
        self,
        emb_seq: torch.Tensor,      # [N, T, d] 已組好的嵌入（[SOS]+ctx+name+...），尚未加 positional
        token_ids: torch.Tensor,    # [N, T] 只用來用 argmax 找 EOT 位置
        add_positional: bool = True,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        assert emb_seq.ndim == 3 and token_ids.ndim == 2
        assert emb_seq.shape[:2] == token_ids.shape, "shape mismatch between emb_seq and token_ids"
        N, T, D = emb_seq.shape
        assert T <= self.context_length, f"sequence too long: {T} > {self.context_length}"

        # 1) 加 positional（CLIP 的 pos 是 [context_length, d] 的參數）
        x = emb_seq.to(self.dtype)
        if add_positional and self.positional_embedding is not None:
            pos = self.positional_embedding[:T, :].to(x.dtype)   # [T, d]
            x = x + pos.unsqueeze(0)                              # [N, T, d]

        # 2) 走 Transformer（CLIP 要 [L, N, D]）
        x = x.permute(1, 0, 2)           # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)           # LND -> NLD

        # 3) LayerNorm
        x = self.ln_final(x).to(dtype)

        # 4) 取 EOT 位置（CoOp 慣例：token_ids.argmax(dim=-1)）
        eot_idx = token_ids.argmax(dim=-1)     # [N]
        x = x[torch.arange(N), eot_idx]        # [N, d]

        # 5) 線性投影到文本特徵空間
        x = x @ self.text_projection           # [N, proj]

        # 6) L2 normalize（照 CLIP/CoOp 推理習慣）
        if normalize:
            x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x
    @smart_inference_mode()
    def encode_image(self, images, dtype=torch.float32):
        feats = self.model.encode_image(images).to(dtype)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats
class MobileCLIP(TextModel):
    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size, device):
        super().__init__()
        config = self.config_size_map[size]
        # 取回 model 與 tokenizer
        self.model, _, preprocess = mobileclip.create_model_and_transforms(
            f"mobileclip_{config}",
            pretrained=f"mobileclip_{size}.pt",
            device=device,
        )
        self._preprocess = preprocess
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.device = device
        self.eval()
        self.to(device)

        # 暴露給 PromptLearner / 上層使用的關鍵屬性（與 CLIP wrapper 對齊）
        te = self.model.text_encoder
        self.token_embedding = te.embedding_layer               # nn.Embedding
        self.positional_embedding = te.positional_embedding      # PositionalEmbedding 模組
        self.transformer = te.transformer
        self.final_layer_norm = te.final_layer_norm
        self.projection_layer = te.projection_layer              # [d_model, proj]
        self.causal_masking = te.causal_masking
        self.build_attention_mask = te.build_attention_mask
        # 有些實作能取得最大長度（若沒有就保持 None，不硬截斷）
        self.context_length = getattr(self.positional_embedding, "num_embeddings", None)
        self.dtype = torch.float32  # MobileCLIP 多半跑 fp32；需要可改

    def tokenize(self, texts):
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, token_ids, dtype=torch.float32):
        feats = self.model.encode_text(token_ids).to(dtype)
        feats /= feats.norm(p=2, dim=-1, keepdim=True)
        return feats

    # === 關鍵：從 PromptLearner 的 embeddings 直接走完整 text encoder（CoOp 風格）===
    def encode_from_embeddings(
        self,
        emb_seq: torch.Tensor,          # [N, T, d]（純 token embeddings，尚未加 positional）
        token_ids: torch.Tensor = None, # [N, T]（用來取 EOT，argmax）
        use_eot: bool = True,
        add_positional: bool = True,
        apply_dropout: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # 如果底層已經提供同名 API，直接用（你前面在 TextTransformer 也加了）
        te = self.model.text_encoder
        if hasattr(te, "encode_from_embeddings"):
            out = te.encode_from_embeddings(
                emb_seq=emb_seq.to(self.device),
                token_ids=token_ids,
                key_padding_mask=None,
                use_eot=use_eot,
                add_positional=add_positional,
                normalize=False,          # 與 encode_text 對齊：這裡不做 L2
            )
            return out.to(dtype)

        # 後備：手動重建與 encode_text 完全相同的路徑
        x = emb_seq.to(self.device)

        # 加 positional（與 forward_embedding 對齊）
        if add_positional and self.positional_embedding is not None:
            x = x + self.positional_embedding(x.size(1)).to(x.dtype)

        # dropout（與 forward_embedding 對齊）
        if apply_dropout and hasattr(te, "embedding_dropout"):
            x = te.embedding_dropout(x)

        # causal mask（與 encode_text 對齊）
        attn_mask = None
        key_padding_mask = None
        if self.causal_masking and hasattr(te, "build_attention_mask"):
            attn_mask = self.build_attention_mask(
                context_length=x.shape[1], batch_size=x.shape[0]
            ).to(device=x.device, dtype=x.dtype)
            key_padding_mask = None

        # Transformer 疊層（與 encode_text 相同）
        for layer in self.transformer:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # LN → 取句向量（EOT 或最後一個）→ 投影（與 encode_text 相同）
        x = self.final_layer_norm(x).to(dtype)
        if use_eot and token_ids is not None:
            eot_idx = token_ids.argmax(dim=-1)        # [N]
            x = x[torch.arange(x.size(0)), eot_idx]   # [N, d]
        else:
            x = x[:, -1, :]                            # [N, d]
        x = x @ self.projection_layer                  # [N, proj]

        # 與 encode_text 對齊：這裡不做 L2 normalize（外面再做）
        return x
    @smart_inference_mode()
    def encode_image(self, images, dtype=torch.float32):
        feats = self.model.encode_image(images).to(dtype)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats

def build_text_model(variant, device=None):
    LOGGER.info(f"Build text model {variant}")
    base, size = variant.split(":")
    if base == 'clip':
        return CLIP(size, device)
    elif base == 'mobileclip':
        return MobileCLIP(size, device)
    else:
        print("Variant not found")
        assert(False)