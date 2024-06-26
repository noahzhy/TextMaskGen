import jax
import jax.numpy as jnp
import optax
import tensorflow as tf


# dice bce loss
@jax.jit
def dice_bce_loss(logits, targets, smooth=1e-7, **kwargs):
    # dice loss
    pred = jax.nn.sigmoid(logits).flatten()
    true = targets.flatten()
    intersection = jnp.sum(pred * true)
    union = jnp.sum(pred) + jnp.sum(true)
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    # bce loss
    bce = optax.sigmoid_binary_cross_entropy(logits, targets).mean()
    return bce + dice


# mse loss
@jax.jit
def mse_loss(pred, target):
    # softmax activation
    # pred = jax.nn.softmax(pred, axis=-1)
    # mse loss
    loss = jnp.mean(jnp.square(pred - target))
    return loss


# focal loss
@jax.jit
def focal_loss(pred, target):
    pred = jax.nn.sigmoid(pred)
    pos_mask = jnp.equal(target, 1.0)
    neg_mask = jnp.less(target, 1.0)
    neg_weights = jnp.power(1.0 - target, 4)

    pos_loss = -jnp.log(jnp.clip(pred, 1e-5, 1.0 - 1e-5)) * jnp.power(1.0 - pred, 2.0) * pos_mask
    neg_loss = (
        -jnp.log(jnp.clip(1.0 - pred, 1e-5, 1.0 - 1e-5))
        * jnp.power(pred, 2.0)
        * neg_weights
        * neg_mask
    )

    num_pos = jnp.sum(pos_mask)
    pos_loss = jnp.sum(pos_loss)
    neg_loss = jnp.sum(neg_loss)

    loss = jnp.where(num_pos > 0, (pos_loss + neg_loss) / num_pos, neg_loss)
    return loss


if __name__ == "__main__":
    # test dice bce loss
    pred = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    target = jnp.array([0, 1, 0, 1, 0])
    print(dice_bce_loss(pred, target))

    # test focal loss
    pred = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    target = jnp.array([0, 1, 0, 1, 0])
    print(focal_loss(pred, target))
