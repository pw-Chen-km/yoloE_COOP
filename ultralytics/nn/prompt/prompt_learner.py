import torch
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    """
    CoOp-style Prompt Learner（參數化接口，移除 cfg 依賴）
    - 初始化時建立 learnable context（generic 或 per-class），並以完整句子
      "X X ... X <classname> ." 做 tokenize 與 embedding，拆出 [SOS] 與 suffix。
    - forward 組裝 prompts = [SOS] + learnable_ctx(+optional FiLM) + suffix。
    - 提供 build_name_token_ids / assemble / forward_and_encode 以配合 predict_* 腳本。
    """
    def __init__(
        self,
        classnames,
        text_model,
        num_ctx: int = 16,
        ctx_init: str = None,
        per_class: bool = False,
        class_token_position: str = "end",
        cond_dim: int = 0,
        dtype=None,
        trainable_ctx: bool = True,
        add_period: bool = True,
    ):
        super().__init__()

        # 基本屬性
        self.text_model = text_model
        self.classnames = [name.replace("_", " ") for name in classnames]
        n_cls = len(self.classnames)
        n_ctx = int(num_ctx)

        # 決定 dtype、ctx_dim 與裝置
        tok_emb = self.text_model.token_embedding
        emb_weight = tok_emb.weight
        ctx_dim = emb_weight.shape[1]
        if dtype is None:
            dtype = emb_weight.dtype
        device = emb_weight.device

        # 1) 初始化 learnable context 向量
        # ctx_init 為 None 表示隨機初始化；若為空字串 "" 則視為 n_ctx=0（零模板）
        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            # 使用默認 split（忽略多重空白），並過濾空字元，允許 n_ctx=0
            ctx_tokens = [t for t in ctx_init.split() if t]
            n_ctx = len(ctx_tokens)
            prompt = self.text_model.tokenize(ctx_init)
            with torch.no_grad():
                embedding = tok_emb(prompt).to(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if per_class:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype, device=device)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        if not trainable_ctx:
            self.ctx.requires_grad_(False)

        # 2) 整句建構（字串），供 tokenize/embedding 拆前後綴
        name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        self.add_period = bool(add_period)
        prompts = []
        for name in self.classnames:
            tail = name + ("." if self.add_period else "")
            if prompt_prefix:
                prompts.append((prompt_prefix + " " + tail).strip())
            else:
                prompts.append(tail)

        # 3) 先 tokenize 整句 & token_embedding，一次性抽出 prefix/suffix
        tokenized_prompts = torch.cat([self.text_model.tokenize(p) for p in prompts])
        with torch.no_grad():
            full_emb = tok_emb(tokenized_prompts).to(dtype)  # [N, T, d]

        # [SOS] 在最前面：embedding[:, :1, :]
        # suffix 是把原本的 ctx 位置挖掉後的大尾巴：embedding[:, 1 + n_ctx :, :]
        self.register_buffer("token_prefix", full_emb[:, :1, :])            # [N,1,d]
        self.register_buffer("token_suffix", full_emb[:, 1 + n_ctx:, :])    # [N,*,d]

        # 4) 基本欄位保存
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # 給 encode_from_embeddings 的 EOT 尋找
        self.name_lens = name_lens
        self.class_token_position = class_token_position

        # 5) 不截斷：若能取得 context_length，超長就 raise
        context_length = None
        if hasattr(self.text_model, "context_length"):
            try:
                context_length = int(self.text_model.context_length)  # type: ignore[attr-defined]
            except Exception:
                context_length = None
        else:
            pe = getattr(self.text_model, "positional_embedding", None)
            if isinstance(pe, torch.Tensor):
                context_length = int(pe.shape[0])

        if context_length is not None:
            T = tokenized_prompts.shape[1]
            if T > context_length:
                raise ValueError(
                    f"Prompt too long for text model: T={T} > context_length={context_length}. "
                    f"Reduce num_ctx or shorten class names."
                )

        # Optional: 延後建立條件化模組（第一次使用 cond 時再初始化）
        self.cond_to_film = None  # lazy init on first cond

    def forward(self):
        # ctx 維度對齊到每一類
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [N, n_ctx, d]

        prefix = self.token_prefix  # [N,1,d]
        suffix = self.token_suffix  # [N,*,d]

        # 佈局與原 CoOp 同步
        if self.class_token_position == "end":
            # [SOS] + ctx + (class tokens + '.' + [EOT] + pad)
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                out.append(torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1))
            prompts = torch.cat(out, dim=0)

        elif self.class_token_position == "front":
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1]
                out.append(torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1))
            prompts = torch.cat(out, dim=0)

        else:
            raise ValueError(f"Unknown CLASS_TOKEN_POSITION={self.class_token_position}")

        return prompts  # [N, T, d]（與原 tokenized_prompts 長度一致）

    # 給外部優化器使用的一致命名
    @property
    def context_tokens(self):
        return self.ctx

    @torch.no_grad()
    def build_name_token_ids(self):
        # 僅供除錯/檢查 tokenizer 與 embedding；輸出每個類別名稱加句點的 tokens
        return torch.cat([self.text_model.tokenize(name + ".") for name in self.classnames], dim=0)

    def _apply_conditioning(self, ctx, cond):
        if cond is None:
            return ctx
        # lazy init FiLM: cond -> [gamma,beta] in ctx_dim
        ctx_dim = ctx.shape[-1]
        if self.cond_to_film is None:
            self.cond_to_film = nn.Linear(cond.shape[-1], 2 * ctx_dim, bias=True).to(ctx.device, ctx.dtype)
        gb = self.cond_to_film(cond.to(ctx.dtype))  # [1,2D] or [B,2D]，預期 [1,2D]
        if gb.dim() == 2 and gb.shape[0] > 1:
            gb = gb.mean(dim=0, keepdim=True)
        gamma, beta = torch.chunk(gb.squeeze(0), 2, dim=-1)  # [D], [D]
        # broadcast to ctx shape
        ctx = ctx * (1 + gamma) + beta
        return ctx

    def assemble(self, token_ids=None, cond=None):
        # 應用條件化於 ctx（若提供）
        ctx = self.ctx
        if ctx.dim() == 2:
            # [n_ctx,d] -> [N,n_ctx,d]
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if cond is not None:
            ctx = self._apply_conditioning(ctx, cond)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                out.append(torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1))
            prompts = torch.cat(out, dim=0)
        elif self.class_token_position == "front":
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1]
                out.append(torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1))
            prompts = torch.cat(out, dim=0)
        else:
            raise ValueError(f"Unknown CLASS_TOKEN_POSITION={self.class_token_position}")

        # 回傳組裝後的 embeddings 與對齊的 token_ids（直接沿用初始化時的）
        return prompts, self.tokenized_prompts

    def forward_and_encode(self, cond=None, templates=None, weights=None, normalize: bool = True, dtype: torch.dtype = torch.float32):
        # templates/weights 保留接口（此版不套模板融合）
        emb_seq, token_ids = self.assemble(cond=cond)
        # 依賴 text_model 的 encode_from_embeddings
        feats = self.text_model.encode_from_embeddings(
            emb_seq,
            token_ids=token_ids,
            use_eot=True,
            dtype=dtype,
        )  # 已做 L2 normalize（依模型而定）
        if normalize:
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        return feats