import os, sys, time, glob, random
# load yaml config
import yaml
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from fit import lr_schedule, fit, TrainState, load_ckpt
from model.dataloader import DataLoader
from model.loss import *
from model.unetv3_light import UNetV3 as UNet


@jax.jit
def combine_loss(pred, target, n=16):
    target_char = target[:, :, :, 0:1]
    # target_ord = target[:, :, :, 1:1+n]

    pred_char = pred
    # loss_char = focal_loss(pred_char, target_char)
    # loss_ord = batch_dice_coef(pred_ord, target_ord)
    loss = dice_bce_loss(pred_char, target_char)

    return loss, {
        'loss': loss,
        # 'loss_char': loss_char,
        # 'loss_ord': loss_ord,
    }


cfg = yaml.safe_load(open("config.yaml"))
print(cfg)

key = jax.random.PRNGKey(0)
dir_path = "/Users/haoyu/Documents/datasets/lpr/mini_train"
train_dl = DataLoader(dir_path, cfg["batch_size"], cfg["img_size"])
train_len = len(train_dl)
lr_fn = lr_schedule(cfg["lr"], train_len, cfg["epochs"], cfg["warmup"])


@jax.jit
def train_step(state: TrainState, batch, opt_state):
    imgs, labels = batch
    def loss_fn(params):
        pred, updates = state.apply_fn({
            'params': params,
            'batch_stats': state.batch_stats
        }, imgs, mutable=['batch_stats'], rngs={'dropout': key})

        loss, loss_dict = combine_loss(pred, labels)
        return loss, (loss_dict, updates)

    (_, (loss_dict, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=updates['batch_stats'])
    _, opt_state = state.tx.update(grads, opt_state)
    return state, loss_dict, opt_state


@jax.jit
def predict(state: TrainState, batch):
    img, _, label = batch
    pred_ctc = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
    }, img, train=False)
    return pred_ctc, label


# TODO: implement eval_step
# def eval_step(state: TrainState, batch):
#     pred_ctc, label = jax.jit(predict)(state, batch)
#     label = batch_remove_blank(label)
#     pred = batch_ctc_greedy_decoder(pred_ctc)
#     acc = jnp.mean(jnp.array([1 if jnp.array_equal(
#         l, p) else 0 for l, p in zip(label, pred)]))
#     return state, acc


if __name__ == "__main__":
    # cpu mode
    # jax.config.update("jax_platform_name", "cpu")

    key = jax.random.PRNGKey(0)
    x = jnp.zeros((1, *cfg["img_size"], 3))

    model = UNet(cfg["features"], cfg["features"], training=True)
    var = model.init(key, x)
    params = var['params']
    batch_stats = var['batch_stats']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.nadam)(lr_fn),
    )

    # state = load_ckpt(state, cfg["ckpt"])

    fit(state, train_dl, train_dl,
        train_step=train_step,
        eval_step=None,
        num_epochs=cfg["epochs"],
        # eval_freq=cfg["eval_freq"],
        eval_freq=-1,
        log_name="tinyText",
        log_freq=1,
    )
