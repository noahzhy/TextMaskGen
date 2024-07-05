import jax
from jax import numpy as jnp
from flax import linen as nn


class ConvBlock(nn.Module):
    features: int = 64
    n: int = 2
    training: bool = True

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n):
            x = nn.Conv(self.features,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal()
            )(x)
            x = nn.BatchNorm(use_running_average=not self.training)(x)
            x = nn.relu(x)
        return x


class ConvUpsampleSoftmax(nn.Module):
    features: int = 64
    upsample: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_init=nn.initializers.kaiming_normal()
            )(x)
        if self.upsample > 1:
            x = jax.image.resize(x,
                shape=(x.shape[0], x.shape[1] * self.upsample, x.shape[2] * self.upsample, x.shape[3]),
                method='bilinear'
            )
        x = nn.softmax(x)
        return x


# spatial attention
class SpatialAttention(nn.Module):
    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, x):
        avg_pool = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        max_pool = nn.max_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        x = jnp.concatenate([avg_pool, max_pool], axis=-1)
        x = nn.Conv(self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.sigmoid(x)
        return x


# STN module
class STN(nn.Module):

    @nn.compact
    def __call__(self, x):
        xs = nn.Conv(8, kernel_size=7, strides=1, padding='SAME')(x)
        xs = nn.max_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.relu(xs)
        xs = nn.Conv(10, kernel_size=5, strides=1, padding='SAME')(xs)
        xs = nn.max_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.relu(xs)
        xs = xs.reshape((-1, 90))

        theta = nn.Sequential([
            nn.Dense(32),
            nn.relu,
            nn.Dense(6),
        ])(xs)
        theta = theta.reshape((-1, 2, 3))

        grid = jnp.linalg.inv(jnp.diag(jnp.ones((x.shape[0], 2)) + 1e-5)) @ theta
        x = jnp.contrib.scipy.ndimage.map_coordinates(x, grid, order=1, mode='reflect')
        return x


class Encoder(nn.Module):
    features: int = 64
    # filters = [64, 128, 256, 512, 1024]
    training: bool = True

    @nn.compact
    def __call__(self, x):
        e1 = ConvBlock(features=self.features, training=self.training)(x)
        e1_pool = nn.max_pool(e1, window_shape=(2, 2), strides=(2, 2))

        e2 = ConvBlock(features=self.features * 2, training=self.training)(e1_pool)
        e2_pool = nn.max_pool(e2, window_shape=(2, 2), strides=(2, 2))

        e3 = ConvBlock(features=self.features * 4, training=self.training)(e2_pool)
        e3_pool = nn.max_pool(e3, window_shape=(2, 2), strides=(2, 2))

        e4 = ConvBlock(features=self.features * 8, training=self.training)(e3_pool)
        e4_pool = nn.max_pool(e4, window_shape=(2, 2), strides=(2, 2))

        e5 = ConvBlock(features=self.features * 16, training=self.training)(e4_pool)

        return e1, e2, e3, e4, e5


class Decoder(nn.Module):
    features: int = 64
    ord_nums: int = 16
    training: bool = True

    @nn.compact
    def __call__(self, e1, e2, e3, e4, e5):
        up_chans = self.features * 4

        e3_d4 = nn.max_pool(e3, window_shape=(2, 2), strides=(2, 2))
        e3_d4 = ConvBlock(features=self.features, n=1, training=self.training)(e3_d4)
        e4_d4 = ConvBlock(features=self.features, n=1, training=self.training)(e4)
        e5_d4 = jax.image.resize(e5, shape=(e5.shape[0], e5.shape[1] * 2, e5.shape[2] * 2, e5.shape[3]), method='bilinear')
        e5_d4 = ConvBlock(features=self.features, n=1, training=self.training)(e5_d4)
        d4 = jnp.concatenate([e3_d4, e4_d4, e5_d4], axis=-1)
        d4 = ConvBlock(features=up_chans, n=1, training=self.training)(d4)

        e2_d3 = nn.max_pool(e2, window_shape=(2, 2), strides=(2, 2))
        e2_d3 = ConvBlock(features=self.features, n=1, training=self.training)(e2_d3)
        e3_d3 = ConvBlock(features=self.features, n=1, training=self.training)(e3)
        e4_d3 = jax.image.resize(e4, shape=(e4.shape[0], e4.shape[1] * 2, e4.shape[2] * 2, e4.shape[3]), method='bilinear')
        e4_d3 = ConvBlock(features=self.features, n=1, training=self.training)(e4_d3)
        d3 = jnp.concatenate([e2_d3, e3_d3, e4_d3], axis=-1)
        d3 = ConvBlock(features=up_chans, n=1, training=self.training)(d3)

        e1_d2 = nn.max_pool(e1, window_shape=(2, 2), strides=(2, 2))
        e1_d2 = ConvBlock(features=self.features, n=1, training=self.training)(e1_d2)
        e2_d2 = ConvBlock(features=self.features, n=1, training=self.training)(e2)
        d3_d2 = jax.image.resize(d3, shape=(d3.shape[0], d3.shape[1] * 2, d3.shape[2] * 2, d3.shape[3]), method='bilinear')
        d3_d2 = ConvBlock(features=self.features, n=1, training=self.training)(d3_d2)
        d2 = jnp.concatenate([e1_d2, e2_d2, d3_d2], axis=-1)
        d2 = ConvBlock(features=up_chans, n=1, training=self.training)(d2)

        e1_d1 = ConvBlock(features=self.features, n=1, training=self.training)(e1)
        d2_d1 = jax.image.resize(d2, shape=(d2.shape[0], d2.shape[1] * 2, d2.shape[2] * 2, d2.shape[3]), method='bilinear')
        d2_d1 = ConvBlock(features=self.features, n=1, training=self.training)(d2_d1)
        d3_d1 = jax.image.resize(d3, shape=(d3.shape[0], d3.shape[1] * 4, d3.shape[2] * 4, d3.shape[3]), method='bilinear')
        d3_d1 = ConvBlock(features=self.features, n=1, training=self.training)(d3_d1)
        d4_d1 = jax.image.resize(d4, shape=(d4.shape[0], d4.shape[1] * 8, d4.shape[2] * 8, d4.shape[3]), method='bilinear')
        d4_d1 = ConvBlock(features=self.features, n=1, training=self.training)(d4_d1)
        d1 = jnp.concatenate([e1_d1, d2_d1, d3_d1, d4_d1], axis=-1)
        d1 = ConvBlock(features=up_chans, n=1, training=self.training)(d1)

        # branch for charmap
        char = ConvBlock(features=self.features, n=1, training=self.training)(d1)
        char = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            strides=1,
            kernel_init=nn.initializers.kaiming_normal(),
        )(char)

        hmap = ConvBlock(features=self.features, n=1, training=self.training)(d1)
        hmap = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            strides=1,
            kernel_init=nn.initializers.kaiming_normal(),
        )(hmap)

        # supervision
        d1 = ConvUpsampleSoftmax(features=self.ord_nums, upsample=0)(d1)

        if self.training:
            d2 = ConvUpsampleSoftmax(features=self.ord_nums, upsample=2)(d2)
            d3 = ConvUpsampleSoftmax(features=self.ord_nums, upsample=4)(d3)
            d4 = ConvUpsampleSoftmax(features=self.ord_nums, upsample=8)(d4)
            d5 = ConvUpsampleSoftmax(features=self.ord_nums, upsample=16)(e5)
            return hmap, char, (d1, d2, d3, d4, d5)

        return hmap, char, d1


class UNetV3(nn.Module):
    features: int = 32
    ord_nums: int = 16
    training: bool = True

    @nn.compact
    def __call__(self, x):
        z1, z2, z3, z4, z5 = Encoder(
            features=self.features,
            training=self.training,
        )(x)
        y = Decoder(
            features=self.features,
            ord_nums=self.ord_nums,
            training=self.training,
        )(z1, z2, z3, z4, z5)
        return y


if __name__ == '__main__':
    # jax cpu
    jax.config.update("jax_platform_name", "cpu")

    key = jax.random.PRNGKey(0)
    model = UNetV3(16, training=True)
    x = jnp.ones((1, 256, 256, 3))
    params = model.init(key, x)
    out, batch_stats = model.apply(
        params, x,
        mutable=['batch_stats'],
        rngs={'dropout': key}
    )

    table_fn = nn.tabulate(
        model,
        key,
        compute_flops=True,
        compute_vjp_flops=True,
    )
    print(table_fn(x))

    for y in out:
        print(y.shape)
