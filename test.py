import os, sys, time, glob, random
# load yaml config
import yaml
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from PIL import Image

from fit import lr_schedule, fit, TrainState, load_ckpt
from model.model import UNet


key = jax.random.PRNGKey(0)
cfg = yaml.safe_load(open("config.yaml"))
print(cfg)


# predict one image
def predict_one_image(state, img_path, cfg):
    img = jnp.array(Image.open(img_path).convert("RGB").resize((cfg["img_size"], cfg["img_size"])))
    # expand dims
    img = jnp.expand_dims(img, axis=0)
    img = img / 255.0
    pred = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats,
    }, img, mutable=['batch_stats'], rngs={'dropout': key})
    return pred


if __name__ == "__main__":
    x = jnp.zeros((1, cfg["img_size"], cfg["img_size"], 3))

    model = UNet(cfg["features"], training=False)
    var = model.init(key, x)
    params = var['params']
    batch_stats = var['batch_stats']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.nadam)(0),
    )

    state = load_ckpt(state, cfg["ckpt"])

    # predict one image
    img_path = '/Users/haoyu/Desktop/world license plate builder/output/image/*.png'
    img_path = random.choice(glob.glob(img_path))
    (hmap, pixs, ord_), _ = predict_one_image(state, img_path, cfg)
    # print(hmap.shape, char.shape, ord_.shape)
    # sigmoid activation
    hmap = jax.nn.sigmoid(hmap)
    pixs = jax.nn.sigmoid(pixs)
    ord_ = jax.nn.sigmoid(ord_)
    # softmax activation
    # ord_ = jax.nn.softmax(ord_, axis=-1)
    # argmax
    ord_ = jnp.argmax(ord_, axis=-1)

    # draw hmap and ordmap via plt
    from matplotlib import pyplot as plt
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    axis[0].imshow(hmap[0, :, :, :], cmap='gray')
    axis[0].set_title('heatmap')
    axis[0].axis('off')
    axis[1].imshow(pixs[0, :, :, :], cmap='gray')
    axis[1].set_title('pixel map')
    axis[1].axis('off')
    # axis[2].imshow(jnp.max(ord_[0, :, :, :], axis=-1), cmap='gray')

    # # show each ordmap channel
    # for i in range(ord_.shape[-1]):
    #     plt.figure()
    #     plt.imshow(ord_[0, :, :, i], cmap='gray')
    #     plt.title(f'ordmap_{i}')
    #     plt.axis('off')

    axis[2].imshow(ord_[0, :, :], cmap='gray')
    axis[2].set_title('ordermap')
    axis[2].axis('off')
    plt.show()
