import os, sys, random, time, glob

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import io as tfio
from tensorflow import image as tfim
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# jax data loader
class DataLoader:
    def __init__(self, root_dir, batch_size, image_size, shuffle=True, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.shuffle = shuffle
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = glob.glob(os.path.join(self.root_dir, '*.png'))

    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __iter__(self):
        if self.shuffle: random.shuffle(self.image_paths)

        bs = self.batch_size
        for i in range(0, len(self.image_paths), bs):
            images, labels = [], []
            for image_path in self.image_paths[i:i + bs]:
                image, label = self.read_image(image_path)
                images.append(image)
                labels.append(label)
            yield np.array(images), np.array(labels)

    def read_image(self, image_path):
        image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0

        npy_f = image_path.replace('.png', '.npy')
        labels = np.load(npy_f).astype(np.float32)
        return image, labels


if __name__ == "__main__":
    PATH = '/Users/haoyu/Desktop/world license plate builder/output/image'
    BATCH_SIZE = 8
    IMAGE_SIZE = 256

    train_data = DataLoader(PATH, BATCH_SIZE, IMAGE_SIZE)

    from matplotlib import pyplot as plt
    from PIL import Image, ImageDraw

    for idx, data in enumerate(train_data):
        x, y = data
        target_hmap = y[:, :, :, :1]
        target_bbox = y[:, :, :, 1:5]
        target_reg = y[:, :, :, 5:6]
        target_ord = y[:, :, :, 6:]
        ordmap = np.max(target_ord, axis=-1)
        # plt.imshow(ordmap[0], cmap='gray')
        # plt.show()
        fig, axis = plt.subplots(1, 3, figsize=(15, 5))
        axis[0].imshow(x[0])
        axis[0].set_title('image')
        axis[0].axis('off')
        axis[1].imshow(target_hmap[0, :, :, 0], cmap='gray')
        axis[1].set_title('heatmap')
        axis[1].axis('off')
        axis[2].imshow(ordmap[0], cmap='gray')
        axis[2].set_title('ordmap')
        axis[2].axis('off')
        plt.show()
        break