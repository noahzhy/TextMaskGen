import os, sys, random, time, glob

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import io as tfio
from tensorflow import image as tfim
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# resize and keep the aspect ratio via pillow
def resize(image, width=256, height=256, inter=Image.Resampling.LANCZOS) -> np.ndarray:
    if width is not None:
        wpercent = (width / float(image.size[0]))
        hsize = int(image.size[1] * wpercent)
        image = image.resize((width, hsize), inter)

    # if height is not None:
    #     hpercent = (height / float(image.size[1]))
    #     wsize = int(image.size[0] * hpercent)
    #     image = image.resize((wsize, height), inter)

    black = Image.new(image.mode, (width, height), 0 if image.mode == 'L' else (0, 0, 0))
    black.paste(image, (0, 0))
    black = np.array(black)

    if image.mode == 'L': black = np.expand_dims(black, axis=-1)

    return black


class DataLoader:
    def __init__(self, root_dir, batch_size, image_size=(128,256), shuffle=False, seed=2024, augment=True):
        random.seed(seed)
        np.random.seed(seed)
        self.shuffle = shuffle
        self.augment = augment
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size # (H, W)
        self.image_paths = glob.glob(os.path.join(self.root_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __iter__(self):
        if self.shuffle: random.shuffle(self.image_paths)

        bs = self.batch_size
        for i in range(0, len(self.image_paths), bs):
            images, labels, ordmaps= [], [], []
            for image_path in self.image_paths[i:i + bs]:
                image, label, ordmap = self.read_image(image_path)

                if self.augment:
                    if random.random() > 0.5:
                        image = 1.0 - image

                images.append(image)
                labels.append(label)
                ordmaps.append(ordmap)
            yield np.array(images), np.array(labels), np.array(ordmaps)

    def read_image(self, image_path):
        h, w = self.image_size
        image = Image.open(image_path).convert('RGB')
        image = resize(image, w, h) / 255.0

        npy_f = image_path.replace('.jpg', '.npy')
        labels = np.load(npy_f, allow_pickle=True).astype(np.uint8)
        labels = Image.fromarray(labels).convert('L')
        labels = resize(labels, w, h, Image.Resampling.NEAREST)
        labels = np.array(labels, dtype=np.uint8)

        labels_raw = labels
        # to one-hot labels
        ordmap = np.zeros((h, w, 16), dtype=np.float32)
        uni = np.unique(labels)
        for idx, i in enumerate(uni[1:17]):
            ordmap[..., idx] = np.squeeze(np.where(labels_raw == i, 1, 0), axis=-1)

        ordmap = np.array(ordmap, dtype=np.float32)
        # where > 0 is mask
        labels = np.max(np.where(labels > 0, 1, 0), axis=-1)
        labels = np.expand_dims(labels, axis=-1)
        return image, labels, ordmap


if __name__ == "__main__":
    PATH = '/Users/haoyu/Documents/datasets/lpr/mini_train'
    BATCH_SIZE = 12
    IMAGE_SIZE = (128, 256)

    train_data = DataLoader(PATH, BATCH_SIZE, IMAGE_SIZE)

    from matplotlib import pyplot as plt
    from PIL import Image, ImageDraw

    for idx, data in enumerate(train_data):
        image, label, ordmap = data
        print(image.shape, label.shape, ordmap.shape)
        fig = plt.figure(figsize=(15, 5))
        for i, data in enumerate([image[0], label[0], ordmap[0]]):
            ax = fig.add_subplot(1, 3, i + 1)
            if data.shape[-1] == 1:
                ax.imshow(data)
            else:
                ax.imshow(np.max(data[:,:,:5], axis=-1))
        plt.show()
        break