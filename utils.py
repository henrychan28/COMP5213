import os
import pickle

import tensorflow as tf
from PIL import Image
import numpy as np
import os

def read_pictures_from_directory_as_numpy(directory, file_type="jpg"):
    images = np.empty((0, 218, 178, 3))
    for file in os.listdir(directory):
        if file.endswith(".{0}".format(file_type)):
            file_directory = os.path.join(directory, file)
            image = np.array(Image.open(file_directory, 'r'))
            images = np.append(images, [image], axis=0)
    return images.astype(np.uint8)

def convert_numpy_images_to_tensors_with_rescaling(images, image_size=64, worker_device="/gpu:0"):
    with tf.device(worker_device):
        images = tf.convert_to_tensor(images, tf.float32)
        images = images[:, 40:188, 15:163, :]
        x_in = tf.image.resize_images(
            images, [image_size, image_size], method=0, align_corners=False)
        x_in = (tf.cast(x_in, tf.float32)
                + tf.random_uniform(tf.shape(x_in))) / 256.
    return x_in


def load_latent_space_samples(directory, worker_device="/gpu:0"):
    with tf.device(worker_device):
        with open(directory, 'r') as f:
            latent_space_samples = pickle.load(f)
        return tf.convert_to_tensor(latent_space_samples)

def checkfile(path):
    path      = os.path.expanduser(path)

    if not os.path.exists(path):
        return path

    root, ext = os.path.splitext(os.path.expanduser(path))
    dir       = os.path.dirname(root)
    fname     = os.path.basename(root)
    candidate = fname+ext
    index     = 0
    ls        = set(os.listdir(dir))
    while candidate in ls:
        candidate = "{}{}{}".format(fname,index,ext)
        index    += 1
    return os.path.join(dir,candidate)    

def save_numpy_array_to_directory(array, filename, directory):
    file_directory = checkfile(directory + "/" + filename)
    import scipy.misc
    scipy.misc.imsave(file_directory, array)
        