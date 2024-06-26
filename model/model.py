import jax
from jax import numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, x):
        z1 = nn.Conv(self.features, kernel_size=(3, 3))(x)
        z1 = nn.relu(z1)
        z1 = nn.Conv(self.features, kernel_size=(3, 3))(z1)
        z1 = nn.BatchNorm(use_running_average=not self.training)(z1)
        z1 = nn.relu(z1)
        z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

        z2 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
        z2 = nn.relu(z2)
        z2 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
        z2 = nn.BatchNorm(use_running_average=not self.training)(z2)
        z2 = nn.relu(z2)
        z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
        z3 = nn.relu(z3)
        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        z3 = nn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = nn.relu(z3)
        z3_pool = nn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

        z4 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z3_pool)
        z4 = nn.relu(z4)
        z4 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z4)
        z4 = nn.BatchNorm(use_running_average=not self.training)(z4)
        z4 = nn.relu(z4)
        z4_dropout = nn.Dropout(0.5, deterministic=False)(z4)
        z4_pool = nn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

        z5 = nn.Conv(self.features * 16, kernel_size=(3, 3))(z4_pool)
        z5 = nn.relu(z5)
        z5 = nn.Conv(self.features * 16, kernel_size=(3, 3))(z5)
        z5 = nn.BatchNorm(use_running_average=not self.training)(z5)
        z5 = nn.relu(z5)
        z5_dropout = nn.Dropout(0.5, deterministic=False)(z5)

        return z1, z2, z3, z4_dropout, z5_dropout


class Decoder(nn.Module):
    features: int = 64
    ord_nums: int = 16
    training: bool = True

    @nn.compact
    def __call__(self, z1, z2, z3, z4, z5):
        z6_up = jax.image.resize(z5, shape=(z5.shape[0], z5.shape[1] * 2, z5.shape[2] * 2, z5.shape[3]),
                                 method='nearest')
        z6 = nn.Conv(self.features * 8, kernel_size=(2, 2))(z6_up)
        z6 = nn.relu(z6)
        z6 = jnp.concatenate([z4, z6], axis=3)
        z6 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = nn.relu(z6)
        z6 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = nn.BatchNorm(use_running_average=not self.training)(z6)
        z6 = nn.relu(z6)

        z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
                                 method='nearest')
        z7 = nn.Conv(self.features * 4, kernel_size=(2, 2))(z7_up)
        z7 = nn.relu(z7)
        z7 = jnp.concatenate([z3, z7], axis=3)
        z7 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = nn.relu(z7)
        z7 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = nn.BatchNorm(use_running_average=not self.training)(z7)
        z7 = nn.relu(z7)

        z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
                                 method='nearest')
        z8 = nn.Conv(self.features * 2, kernel_size=(2, 2))(z8_up)
        z8 = nn.relu(z8)
        z8 = jnp.concatenate([z2, z8], axis=3)
        z8 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = nn.relu(z8)
        z8 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = nn.BatchNorm(use_running_average=not self.training)(z8)
        z8 = nn.relu(z8)

        z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
                                 method='nearest')
        z9 = nn.Conv(self.features, kernel_size=(2, 2))(z9_up)
        z9 = nn.relu(z9)
        z9 = jnp.concatenate([z1, z9], axis=3)
        z9 = nn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = nn.relu(z9)

        # branch for heatmap
        obj = nn.Conv(self.features, kernel_size=(3, 3))(z9)
        obj = nn.BatchNorm(use_running_average=not self.training)(obj)
        obj = nn.relu(obj)
        obj = nn.Conv(1, kernel_size=(1, 1))(obj)

        char = nn.Conv(self.features, kernel_size=(3, 3))(z9)
        char = nn.BatchNorm(use_running_average=not self.training)(char)
        char = nn.relu(char)
        char = nn.Conv(1, kernel_size=(1, 1))(char)

        # # skip connection to z5 (bs, 16, 16, 256)
        skip = nn.Conv(128, kernel_size=(1, 1))(z5)
        skip = nn.relu(skip)
        skip = jax.image.resize(skip, shape=(skip.shape[0], skip.shape[1] * 16, skip.shape[2] * 16, skip.shape[3]), method='nearest')

        # concat
        # z9 = jnp.concatenate([z9, skip], axis=-1)

        # z5 upsample 4 times
        # up = UpSample(up_repeat=4, n_channels=32)(z5, train=self.training)
        # concat z9 and up
        z9 = jnp.concatenate([skip, z9], axis=-1)
        # print(z9.shape)

        # branch for ordmap
        ord_ = nn.Conv(64, kernel_size=(3, 3), strides=1)(z9)
        ord_ = nn.BatchNorm(use_running_average=not self.training)(ord_)
        ord_ = nn.softmax(ord_) * char
        ord_ = nn.Conv(
            features=self.ord_nums,
            kernel_size=(1, 1),
            strides=1,
            kernel_init=nn.initializers.kaiming_normal(),
            use_bias=True,
        )(ord_)

        return obj, char, ord_


class UpSample(nn.Module):
    up_repeat: int = 3
    n_channels: int = 128

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.up_repeat):
            x = nn.Conv(
                features=self.n_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_init=nn.initializers.kaiming_normal(),
                use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.PReLU()(x)
            x = jax.image.resize(x,
                shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
                method="nearest")           
        return x


# spatial attention
class SpatialAttention(nn.Module):
    features: int = 64

    @nn.compact
    def __call__(self, x):
        avg_pool = jnp.mean(x, axis=-1, keepdims=True)
        max_pool = jnp.max(x, axis=-1, keepdims=True)
        pool = jnp.concatenate([avg_pool, max_pool], axis=-1)
        pool = nn.Conv(self.features, kernel_size=(1, 1))(pool)
        pool = nn.relu(pool)
        pool = nn.Conv(1, kernel_size=(1, 1))(pool)
        pool = nn.sigmoid(pool)
        return pool


# channel attention
class ChannelAttention(nn.Module):
    features: int = 64

    @nn.compact
    def __call__(self, x):
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True)
        max_pool = jnp.max(x, axis=(1, 2), keepdims=True)
        pool = jnp.concatenate([avg_pool, max_pool], axis=-1)
        pool = nn.Conv(128, kernel_size=(1, 1))(pool)
        pool = nn.relu(pool)
        pool = nn.Conv(x.shape[-1], kernel_size=(1, 1))(pool)
        pool = nn.sigmoid(pool)
        return x * pool


class UNet(nn.Module):
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
    model = UNet(16, training=True)
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
