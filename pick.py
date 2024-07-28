import os, sys, random, glob, time


# random copy 128 files from src to dst
def pick(src, dst, num=128):
    files = glob.glob(src + '/*.jpg')
    random.shuffle(files)
    for i in range(num):
        os.system('cp %s %s' % (files[i], dst))


if __name__ == '__main__':
    src = "/Users/haoyu/Documents/datasets/lpr/train"
    dst = "/Users/haoyu/Documents/datasets/lpr/mini_train"
    num = 256
    pick(src, dst, num)
    print('done')
