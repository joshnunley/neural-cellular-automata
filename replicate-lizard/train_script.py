# Packages
import numpy as np
import tensorflow as tf

# STD Lib
import subprocess

# Local
from model import CAModel
from train_utils import (
    SamplePool,
    make_circle_masks,
    generate_pool_figures,
    visualize_batch,
    plot_loss,
    export_model,
)
from utils import imshow, load_emoji, zoom, to_rgb, to_rgba
from constants import (
    TARGET_PADDING,
    TARGET_EMOJI,
    POOL_SIZE,
    BATCH_SIZE,
    USE_PATTERN_POOL,
    CHANNEL_N,
    DAMAGE_N,
)

target_img = load_emoji(TARGET_EMOJI)
# imshow(zoom(to_rgb(target_img), 2), fmt="png")

p = TARGET_PADDING
pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
seed = np.zeros([h, w, CHANNEL_N], np.float32)
seed[h // 2, w // 2, 3:] = 1.0


def loss_f(x):
    return tf.reduce_mean(tf.square(to_rgba(x) - pad_target), [-2, -3, -1])


ca = CAModel()

loss_log = []

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

loss0 = loss_f(seed).numpy()
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

remove_old_directory = "rm -rf train_log/*"
subprocess.run(
    remove_old_directory.split(),
    cwd="/N/u/joshnunl/BigRed200/neural-cellular-automata/replicate-lizard",
)
make_new_directory = "mkdir -p train_log"
subprocess.run(
    make_new_directory.split(),
    cwd="/N/u/joshnunl/BigRed200/neural-cellular-automata/replicate-lizard",
)


@tf.function
def train_step(x):
    iter_n = tf.random.uniform([], 64, 96, tf.int32)
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            x = ca(x)
        loss = tf.reduce_mean(loss_f(x))
    grads = g.gradient(loss, ca.weights)
    grads = [g / (tf.norm(g) + 1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss


for i in range(8000 + 1):
    if USE_PATTERN_POOL:
        batch = pool.sample(BATCH_SIZE)
        x0 = batch.x
        loss_rank = loss_f(x0).numpy().argsort()[::-1]
        x0 = x0[loss_rank]
        x0[:1] = seed
        if DAMAGE_N:
            damage = 1.0 - make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
            x0[-DAMAGE_N:] *= damage
    else:
        x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)

    x, loss = train_step(x0)

    if USE_PATTERN_POOL:
        batch.x[:] = x
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    # if step_i % 10 == 0:
    #    generate_pool_figures(pool, step_i)
    if step_i % 100 == 0:
        #    visualize_batch(x0, x, step_i)
        plot_loss(loss_log)
        export_model(ca, "train_log/%04d" % step_i)

    print("\r step: %d, log10(loss): %.3f" % (len(loss_log), np.log10(loss)), end="")
