import jax
from jax import numpy as jnp
from flax import linen as nn


class AddCoords(nn.Module):
    with_r: bool = True
    with_boundary: bool = False
    heatmap: bool = None

    @nn.compact
    def __call__(self, x):
        bs, h, w, c = x.shape

        xx_ones = jnp.ones((1, h), dtype=jnp.int32).reshape(1, h, 1)
        xx_range = jnp.arange(w, dtype=jnp.int32).reshape(1, 1, w)
        xx_channel = jnp.matmul(xx_ones.astype(jnp.float32), xx_range.astype(jnp.float32))
        xx_channel = jnp.expand_dims(xx_channel, -1)

        yy_ones = jnp.ones((1, w), dtype=jnp.int32).reshape(1, 1, w)
        yy_range = jnp.arange(h, dtype=jnp.int32).reshape(1, h, 1)
        yy_channel = jnp.matmul(yy_range.astype(jnp.float32), yy_ones.astype(jnp.float32))
        yy_channel = jnp.expand_dims(yy_channel, -1)

        xx_channel = (xx_channel / (w - 1)) * 2 - 1
        yy_channel = (yy_channel / (h - 1)) * 2 - 1

        xx_channel = jnp.repeat(xx_channel, bs, axis=0)
        yy_channel = jnp.repeat(yy_channel, bs, axis=0)
        ret = jnp.concatenate([x, xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = jnp.sqrt(jnp.square(xx_channel) + jnp.square(yy_channel))
            rr /= jnp.max(rr)
            ret = jnp.concatenate([ret, rr], axis=-1)

        if self.with_boundary and self.heatmap is not None:
            boundary = jnp.zeros_like(self.heatmap)
            boundary = jnp.expand_dims(boundary, axis=-1)
            ret = jnp.concatenate([ret, boundary], axis=-1)

        return ret


class CoordConv(nn.Module):
    features: int
    with_r: bool = True

    @nn.compact
    def __call__(self, x):
        x = AddCoords(with_r=self.with_r)(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        return x


if __name__ == '__main__':
    # jax cpu
    jax.config.update("jax_platform_name", "cpu")

    key = jax.random.PRNGKey(0)
    model = CoordConv(features=3, with_r=True)
    x = jnp.ones((1, 64, 64, 3))
    params = model.init(key, x)
    out, batch_stats = model.apply(
        params, x,
        mutable=['batch_stats'],
        rngs={'dropout': key}
    )

    table_fn = nn.tabulate(
        model, key,
        # compute_flops=True,
        # compute_vjp_flops=True,
    )
    print(table_fn(x))

    for y in out:
        print(y.shape)
