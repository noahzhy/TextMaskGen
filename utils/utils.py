# import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


def nms(heat, kernel=3):
    hmax = nn.max_pool(heat, (kernel, kernel), (1, 1), "SAME")
    keep = jnp.equal(heat, hmax)
    return heat * keep


def topk(hm, k=100):
    batch, height, width = jnp.shape(hm)[0], jnp.shape(hm)[1], jnp.shape(hm)[2]
    scores = jnp.reshape(hm, (batch, -1))
    topk_scores, topk_inds = jax.lax.top_k(scores, k=k)

    topk_xs = topk_inds // 1 % width
    topk_ys = topk_inds // 1 // width
    topk_inds = topk_ys * width + topk_xs
    return topk_scores, topk_inds, topk_ys, topk_xs


def decode(heat, wh, k=100):
    batch, height, width = jnp.shape(heat)[0], jnp.shape(heat)[1], jnp.shape(heat)[2]
    heat = nms(heat)
    scores, inds, ys, xs = topk(heat, k=k)

    ys = jnp.expand_dims(ys, axis=-1)
    xs = jnp.expand_dims(xs, axis=-1)

    wh = jnp.reshape(wh, (batch, -1, jnp.shape(wh)[-1]))
    wh = jnp.take_along_axis(wh, jnp.expand_dims(inds, axis=-1), axis=1)

    scores = jnp.expand_dims(scores, axis=-1)

    ymin = ys - wh[..., 0:1]
    xmin = xs - wh[..., 1:2]
    ymax = ys + wh[..., 2:3]
    xmax = xs + wh[..., 3:4]

    bboxes = jnp.concatenate([ymin, xmin, ymax, xmax], axis=-1)
    detections = jnp.concatenate([bboxes, scores], axis=-1)
    return detections


if __name__ == "__main__":
    heat = jnp.zeros((1, 128, 128, 1))
    wh = jnp.zeros((1, 128, 128, 4))
    detections = decode(heat, wh)
    print(detections)
