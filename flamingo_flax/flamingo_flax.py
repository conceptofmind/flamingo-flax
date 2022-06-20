import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

class FeedForward(nn.Module):
    dim: int
    mult: int = 4

    @nn.compact
    def __call__(self, x):
        inner_dim = int(self.dim * self.mult)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features = inner_dim, use_bias = False)(x)
        x = nn.gelu(x)
        x = nn.Dense(features = self.dim, use_bias = False)(x)
        return x

class PerceiverAttention(nn.Module):
    dim: int
    dim_head: int = 64
    heads: int = 8

    @nn.compact
    def forward(self, x, latents):

        scale = self.dim_head ** -0.5
        heads = self.heads
        inner_dim = self.dim_head * heads

        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = nn.LayerNorm(dim)(x)
        latents = nn.LayerNorm(dim)(latents)

        b, m, h = *x.shape[:2], self.heads

        q = nn.Linear(dim, inner_dim, bias = False)(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = jnp.concatenate((x, latents), dim = -2)
        k, v = nn.Linear(dim, inner_dim * 2, bias = False)(kv_input).split(2, axis = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return nn.Linear(dim, bias = False)(out)

class PerceiverResampler(nn.Module):
    dim: int
    depth: int
    dim_head: int = 64
    heads: int = 8
    num_latents: int = 64
    num_time_embeds: int = 4
    ff_mult: int = 4

    @nn.compact
    def __call__(self, x):

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))

        layers = []
        for _ in range(self.depth):
            layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.time_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return nn.LayerNorm(dim)(latents)

# gated cross attention

class MaskedCrossAttention(nn.Module):
    dim: int
    dim_head: int = 64
    heads: int = 8
    only_attend_immediate_media: bool = True

    @nn.compact
    def __call__(self, x, media, media_locations = None):

        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        # whether for text to only attend to immediate preceding image, or all images

        self.only_attend_immediate_media = only_attend_immediate_media

        b, t, m = media.shape[:3]
        h = self.heads

        x = nn.LayerNorm(dim)(x)

        q = nn.Linear(inner_dim, bias = False)(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = nn.Linear(inner_dim * 2, bias = False)(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(t, device = x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return nn.Linear(dim, bias = False)(out)

class GatedCrossAttentionBlock(nn.Module):
    dim: int
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    only_attend_immediate_media: bool = True

    @nn.compact
    def __call__(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None  # boolean tensor indicating positions of media - (batch, sequence)
    ):

        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

        x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        return x