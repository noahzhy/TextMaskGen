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


@jax.jit
def dice_coef(pred, target, smooth=1.e-9):
    pred = pred.flatten()
    target = target.flatten()
    transposed_pred = jnp.transpose(pred)
    intersection = jnp.dot(target, transposed_pred)
    union = jnp.dot(target, jnp.transpose(target)) + jnp.dot(pred, jnp.transpose(pred))
    return 1 - (2. * intersection + smooth) / (union + smooth)


@jax.jit
def batch_dice_coef(pred, target):
    dice = 0
    """Dice coeff for batches"""
    for i, p in enumerate(pred):
        dice += dice_coef(p, target)

    return dice / (i + 1)


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

    # dice_coef_loss(
    #     K.theano.shared(np.array([[0,0,0],[0,0,0]])),
    #     K.theano.shared(np.array([[0,0,0],[0,0,0]]))
    # ).eval() # array([ 0.,  0.])

    pred = jnp.array([[0, 0, 0], [1, 1, 1]])
    target = jnp.array([[0, 0, 0]])
    print(batch_dice_coef(pred, target))
