import flax.linen as nn

import math

from functools import wraps

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, repeat
#from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

def masked_fill_(t, mask, value):
    return t * (1 - mask) + value * mask

def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)

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
    def __call__(self, x, latents):

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
        x = nn.LayerNorm(epsilon = 1e-5)(x)
        latents = nn.LayerNorm(epsilon = 1e-5)(latents)

        b, m, h = *x.shape[:2], self.heads

        q = nn.Dense(features = inner_dim, use_bias = False)(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = jnp.concatenate((x, latents), axis = -2)
        k, v = nn.Dense(features = inner_dim * 2, use_bias = False)(kv_input).split(2, axis = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - jnp.amax(sim, axis = -1, keepdims = True)
        attn = nn.softmax(sim, axis = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return nn.Dense(features = self.dim, use_bias = False)(out)

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

        latents = jax.random.normal(key, [self.num_latents, self.dim])
        time_pos_emb = jax.random.normal(key ,[self.num_time_embeds, 1, self.dim])

        layers = []
        for _ in range(self.depth):
            layers.append([
                PerceiverAttention(dim = self.dim, dim_head = self.dim_head, heads = self.heads),
                FeedForward(dim = self.dim, mult = self.ff_mult)
            ])

        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + time_pos_emb[:times]

        latents = repeat(latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return nn.LayerNorm(epsilon = 1e-5)(latents)

# gated cross attention

class MaskedCrossAttention(nn.Module):
    dim: int
    dim_head: int = 64
    heads: int = 8
    only_attend_immediate_media: bool = True

    @nn.compact
    def __call__(self, x, media, media_locations = None):

        scale = self.dim_head ** -0.5
        heads = self.heads
        inner_dim = self.dim_head * heads

        # whether for text to only attend to immediate preceding image, or all images

        only_attend_immediate_media = self.only_attend_immediate_media

        b, t, m = media.shape[:3]
        h = self.heads

        x = nn.LayerNorm(epsilon = 1e-5)(x)

        q = nn.Dense(features = inner_dim, use_bias = False)(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = nn.Dense(features = inner_dim * 2, use_bias = False)(media).split(2, axis = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = jnp.cumsum(media_locations, axis = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = jnp.arange(t) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = jnp.eq if only_attend_immediate_media else jnp.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = masked_fill_(sim, ~text_to_media_mask, -jnp.finfo(sim.dtype).max)

        sim = sim - jnp.amax(sim, axis = -1, keepdims = True)
        attn = nn.softmax(sim, axis = -1)

        if exists(media_locations) and only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            masked_fill_(attn, text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return nn.Dense(features = self.dim, use_bias = False)(out)

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

        attn = MaskedCrossAttention(dim = self.dim, dim_head = self.dim_head, heads = self.heads, only_attend_immediate_media = self.only_attend_immediate_media)
        attn_gate = jnp.array([0.])

        ff = FeedForward(self.dim, mult = self.ff_mult)
        ff_gate = jnp.array([0.])

        x = attn(x, media, media_locations = media_locations) * jnp.tanh(attn_gate) + x
        x = ff(x) * jnp.tanh(ff_gate)  + x
        return x

if __name__ == '__main__':

    import numpy as np

    key = jax.random.PRNGKey(0)

    medias = jax.random.normal(key, (1, 2, 256, 1024))

    perceive = PerceiverResampler(
        dim = 1024,
        depth = 2,
        dim_head = 64,
        heads = 8,
        num_latents = 64,    # the number of latents to shrink your media sequence to, perceiver style
        num_time_embeds = 4  # say you have 4 images maximum in your dialogue
    )
    
    init_rngs = {'params': jax.random.PRNGKey(1), 
                'latents': jax.random.PRNGKey(2), 
                'time_pos_emb': jax.random.PRNGKey(3)}

    params = perceive.init(init_rngs, medias)
    output = perceive.apply(params, medias, rngs=init_rngs)
    print(output.shape)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")