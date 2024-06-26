import os, sys, random, time, glob

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import io as tfio
from tensorflow import image as tfim
from matplotlib import pyplot as plt
# TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# get bbox from mask
def get_bbox(mask_path, target_size=(256, 256)):
    h, w = target_size
    mask = np.array(Image.open(mask_path))
    org_w, org_h, _ = mask.shape
    mask = (mask + 8) // 16

    scale_x = w / org_w
    scale_y = h / org_h

    bboxes = []
    for i in range(1, 17):
        mask_i = mask == i
        if np.sum(mask_i) == 0:
            continue

        rows = np.any(mask_i, axis=1)
        cols = np.any(mask_i, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # numpy to pillow
        x_min = int(cmin * scale_x)
        y_min = int(rmin * scale_y)
        x_max = int(cmax * scale_x)
        y_max = int(rmax * scale_y)
        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes


def bbox_areas_log_np(bbox):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    area = (y_max - y_min + 1) * (x_max - x_min + 1)
    return np.log(area)


def radius_ttf(bbox, h, w):
    alpha = 0.54
    h_radiuses_alpha = int(h / 2.0 * alpha)
    w_radiuses_alpha = int(w / 2.0 * alpha)
    return max(0, h_radiuses_alpha), max(0, w_radiuses_alpha)


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y)

    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[h_radius - top : h_radius + bottom, w_radius - left : w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def draw_heatmaps(shape, bboxes):
    heat_map = np.zeros(shape, dtype=np.float32)

    for b in range(shape[0]):
        for bbox in bboxes[b]:
            bbox = np.asarray(bbox)
            area = bbox_areas_log_np(bbox)
            fake_heatmap = np.zeros((shape[1], shape[2]))
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                # compute heat map
                h_radius, w_radius = radius_ttf(bbox, h, w)
                ct = np.array([
                    (bbox[1] + bbox[3]) / 2,
                    (bbox[0] + bbox[2]) / 2,
                ], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_truncate_gaussian(fake_heatmap, ct_int, h_radius, w_radius)
                heat_map[b, :, :, 0] = np.maximum(heat_map[b, :, :, 0], fake_heatmap)

    return heat_map


# gen ord mask
def gen_ord_mask(mask, n=16):
    w, h, _ = mask.shape
    arr = np.zeros((h, w, n), dtype=np.int32)

    for i in range(n):
        if len(np.where(mask == i+1)[0]) == 0:
            continue

        m = np.where(mask == i+1, 1, 0)
        rmin, rmax = np.where(np.any(m, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(m, axis=0))[0][[0, -1]]

        arr[rmin:rmax+1, cmin:cmax+1, i] = 1
        # arr[:, :, i] = m

    return arr


# a array of shape (H, W, 1) to one-hot array of shape (H, W, C)
def one_hot(array, C):
    C += 1
    H, W, _ = array.shape
    array = np.reshape(array, (H, W))
    array = np.eye(C)[array]
    # exclude background
    array = array[:, :, 1:]
    return array


def preprocess_image(mask_path, image_size=256):
    mask = np.array(
        Image.open(mask_path)
        .convert('L')
        .resize((image_size, image_size), Image.NEAREST)
    )
    mask = (mask + 8) // 16
    mask = np.expand_dims(mask, axis=-1)

    pixel_level = np.where(mask > 0.5, 1, 0)

    bbox = get_bbox(mask_path, target_size=(image_size, image_size))
    heat_map = draw_heatmaps((1, image_size, image_size, 1), [bbox])
    heat_map = np.squeeze(heat_map, axis=0)

    ord_mask = gen_ord_mask(mask)
    # ord_mask = one_hot(mask, 16)

    # # show ord mask
    # plt.imshow(np.max(ord_mask[:, :, :1], axis=-1))
    # plt.show()
    # quit()

    return np.concatenate([heat_map, pixel_level, ord_mask], axis=-1, dtype=np.float32)


if __name__ == "__main__":
    PATH = '/Users/haoyu/Desktop/world license plate builder/output'
    BATCH_SIZE = 8
    IMAGE_SIZE = 256

    mask_path = glob.glob(PATH + '/mask/*.png')

    for f in mask_path:
        d = preprocess_image(f)
        np.save(f.replace('mask', 'image').replace('.png', '.npy'), d)
        print(f.replace('mask', 'image').replace('.png', '.npy'))

    quit()

    heat_map, box_target, reg_weight = draw_heatmaps((1, 256, 256, 1), [bbox])
    # save heatmap
    plt.imshow(heat_map[0, :, :, 0])
    # plt.show()
    plt.imsave('heatmap.png', heat_map[0, :, :, 0])

    # draw bbox via pillow
    from PIL import Image, ImageDraw
    image = Image.open(mask_path[0].replace('mask', 'image')).resize((256, 256))
    for b in bbox:
        ImageDraw.Draw(image).rectangle(b, outline='red')
    # image.show()
    image.save('bbox.png')
