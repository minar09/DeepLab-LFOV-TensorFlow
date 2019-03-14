import os

import tensorflow as tf
from PIL import Image
import numpy as np

# Colour map.
label_colours = [(0, 0, 0),  # 0=Background
                 (128, 0, 0),  # hat
                 (255, 0, 0),  # hair
                 (170, 0, 51),  # sunglasses
                 (255, 85, 0),  # upper-clothes
                 (0, 128, 0),  # skirt
                 (0, 85, 85),  # pants
                 (0, 0, 85),  # dress
                 (0, 85, 0),  # belt
                 (255, 255, 0),  # Left-shoe
                 (255, 170, 0),  # Right-shoe
                 (0, 0, 255),  # face
                 (85, 255, 170),  # left-leg
                 (170, 255, 85),  # right-leg
                 (51, 170, 221),  # left-arm
                 (0, 255, 255),  # right-arm
                 (85, 51, 0),  # bag
                 (52, 86, 128)  # scarf
                 ]


def decode_labels(mask):
    """Decode batch of segmentation masks.

    Args:
      label_batch: result of inference after taking argmax.

    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < 21:
                pixels[k_, j_] = label_colours[k]
    return np.array(img)


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
