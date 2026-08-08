"""Jittor implementation of the JiT-B/16 network.

Parameter names, tensor layouts, and random-initialization rules match the
upstream PyTorch implementation.  The model can therefore either train from
random initialization or load ``model_ema1`` from the official checkpoint.
"""

import math

import jittor as jt
import numpy as np
from jittor import nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    if embed_dim % 2:
        raise ValueError("1-D sine/cosine embedding dimension must be even")
    frequencies = np.arange(embed_dim // 2, dtype=np.float64)
    frequencies /= embed_dim / 2.0
    frequencies = 1.0 / (10000**frequencies)
    angles = np.einsum(
        "m,d->md",
        np.asarray(positions).reshape(-1),
        frequencies,
    )
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Match ``util.model_util.get_2d_sincos_pos_embed`` in the Torch repo."""
    if embed_dim % 2:
        raise ValueError("2-D sine/cosine embedding dimension must be even")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape((2, 1, grid_size, grid_size))
    embedding_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2,
        grid[0],
    )
    embedding_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2,
        grid[1],
    )
    return np.concatenate((embedding_h, embedding_w), axis=1)


def xavier_uniform_flat_(parameter):
    """Apply Xavier uniform after flattening all non-output dimensions."""
    shape = tuple(parameter.shape)
    flattened_shape = (shape[0], int(np.prod(shape[1:])))
    values = jt.init.xavier_uniform(
        flattened_shape,
        dtype=parameter.dtype,
    ).reshape(shape)
    parameter.assign(values)


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rotate_half(x):
    shape = tuple(x.shape)
    paired = x.reshape(shape[:-1] + (shape[-1] // 2, 2))
    rotated = jt.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.reshape(shape)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, seq_len=16, num_cls_token=0, theta=10000.0):
        super().__init__()
        base = np.arange(0, dim, 2, dtype=np.float32)[: dim // 2]
        inv_freq = 1.0 / (theta ** (base / dim))
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = np.einsum("i,j->ij", positions, inv_freq)
        freqs = np.repeat(freqs, 2, axis=-1)

        height = np.broadcast_to(freqs[:, None, :], (seq_len, seq_len, dim))
        width = np.broadcast_to(freqs[None, :, :], (seq_len, seq_len, dim))
        freqs_2d = np.concatenate((height, width), axis=-1).reshape(-1, dim * 2)
        cos = np.cos(freqs_2d).astype(np.float32)
        sin = np.sin(freqs_2d).astype(np.float32)

        if num_cls_token:
            cos = np.concatenate(
                (np.ones((num_cls_token, dim * 2), dtype=np.float32), cos), axis=0
            )
            sin = np.concatenate(
                (np.zeros((num_cls_token, dim * 2), dtype=np.float32), sin), axis=0
            )

        # Leading underscores keep these constants out of state_dict().
        self._freqs_cos = jt.array(cos)
        self._freqs_sin = jt.array(sin)

    def execute(self, x):
        return x * self._freqs_cos + rotate_half(x) * self._freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = jt.ones((hidden_size,), dtype="float32")
        self.variance_epsilon = eps

    def execute(self, hidden_states):
        input_dtype = hidden_states.dtype
        values = hidden_states.float32()
        variance = (values * values).mean(dim=-1, keepdims=True)
        values = values * jt.rsqrt(variance + self.variance_epsilon)
        return (self.weight * values).cast(input_dtype)


class BottleneckPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        pca_dim=128,
        embed_dim=768,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj1 = nn.Conv(
            in_chans,
            pca_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv(pca_dim, embed_dim, kernel_size=1, stride=1, bias=True)

    def execute(self, x):
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size {height}x{width}, expected "
                f"{self.img_size[0]}x{self.img_size[1]}"
            )
        x = self.proj2(self.proj1(x))
        return x.reshape((x.shape[0], x.shape[1], -1)).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jt.exp(
            -math.log(max_period) * jt.arange(half).float32() / float(half)
        )
        args = t.unsqueeze(1).float32() * freqs.unsqueeze(0)
        embedding = jt.concat((jt.cos(args), jt.sin(args)), dim=-1)
        if dim % 2:
            embedding = jt.concat(
                (embedding, jt.zeros_like(embedding[:, :1])), dim=-1
            )
        return embedding

    def execute(self, t):
        embedding = self.timestep_embedding(t, self.frequency_embedding_size)
        embedding = embedding.cast(self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def execute(self, labels):
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def execute(self, x, rope):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            (batch, tokens, 3, self.num_heads, self.head_dim)
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        # The reference implementation performs QK and softmax in fp32.
        attention = jt.matmul(q.float32(), k.float32().transpose(-2, -1))
        attention = nn.softmax(attention * self.scale, dim=-1)
        # Under the reference's outer BF16 autocast, the second matmul is
        # executed in BF16 even though softmax itself is FP32.
        x = jt.matmul(attention.cast(v.dtype), v)
        x = x.transpose(1, 2).reshape((batch, tokens, channels))
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)

    def execute(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(nn.silu(x1) * x2)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def execute(self, x, conditioning):
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def execute(self, x, conditioning, feat_rope):
        chunks = self.adaLN_modulation(conditioning).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class JiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
        )

        self.pos_embed = jt.zeros(
            (1, self.x_embedder.num_patches, hidden_size), dtype="float32"
        )
        self.in_context_posemb = jt.zeros(
            (1, in_context_len, hidden_size), dtype="float32"
        )

        half_head_dim = hidden_size // num_heads // 2
        image_tokens_per_side = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            half_head_dim, image_tokens_per_side, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            half_head_dim,
            image_tokens_per_side,
            num_cls_token=in_context_len,
        )
        self.blocks = nn.ModuleList(
            [
                JiTBlock(hidden_size, num_heads, mlp_ratio=4.0)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """Match the upstream Torch ``JiT.initialize_weights`` routine."""

        def initialize_linear(module):
            if isinstance(module, nn.Linear):
                jt.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    jt.init.zero_(module.bias)

        self.apply(initialize_linear)

        position_embedding = get_2d_sincos_pos_embed(
            int(self.pos_embed.shape[-1]),
            int(self.x_embedder.num_patches**0.5),
        ).astype(np.float32)
        self.pos_embed.assign(jt.array(position_embedding).unsqueeze(0))

        xavier_uniform_flat_(self.x_embedder.proj1.weight)
        xavier_uniform_flat_(self.x_embedder.proj2.weight)
        if self.x_embedder.proj2.bias is not None:
            jt.init.zero_(self.x_embedder.proj2.bias)

        jt.init.gauss_(self.y_embedder.embedding_table.weight, std=0.02)
        jt.init.gauss_(self.t_embedder.mlp[0].weight, std=0.02)
        jt.init.gauss_(self.t_embedder.mlp[2].weight, std=0.02)
        jt.init.gauss_(self.in_context_posemb, std=0.02)

        for block in self.blocks:
            jt.init.zero_(block.adaLN_modulation[-1].weight)
            jt.init.zero_(block.adaLN_modulation[-1].bias)

        jt.init.zero_(self.final_layer.adaLN_modulation[-1].weight)
        jt.init.zero_(self.final_layer.adaLN_modulation[-1].bias)
        jt.init.zero_(self.final_layer.linear.weight)
        jt.init.zero_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        channels = self.out_channels
        side = int(x.shape[1] ** 0.5)
        patch = self.patch_size
        x = x.reshape((x.shape[0], side, side, patch, patch, channels))
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(
            (x.shape[0], channels, side * patch, side * patch)
        )

    def execute(self, x, t, y):
        # torch.amp.autocast casts Conv/Linear inputs to the parameter dtype.
        # Explicitly doing so reproduces that behavior after model.bfloat16().
        x = x.cast(self.pos_embed.dtype)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        conditioning = t_emb + y_emb

        x = self.x_embedder(x) + self.pos_embed
        for index, block in enumerate(self.blocks):
            if index == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(
                    1, self.in_context_len, 1
                )
                x = jt.concat(
                    (context + self.in_context_posemb, x), dim=1
                )
            rope = (
                self.feat_rope
                if index < self.in_context_start
                else self.feat_rope_incontext
            )
            x = block(x, conditioning, rope)

        x = x[:, self.in_context_len :]
        return self.unpatchify(self.final_layer(x, conditioning))


def JiT_B_16(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )
