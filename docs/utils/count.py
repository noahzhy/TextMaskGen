import os
import sys
import time
import glob
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jamo import h2j, j2hcj, j2h
from korean_romanizer.romanizer import Romanizer


def load_dict(dict_path='data/labels.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        _dict = f.read().splitlines()
    _dict = {h2j(_dict[i]): i for i in range(len(_dict))}
    return _dict


label_dict = load_dict('data/labels.names')
print(label_dict)

# split label
# e.g. '63루3348' -> ['6', '3', '루', '3', '3', '4', '8']
# e.g. '서울12가1234' -> ['서울', '1', '2', '가', '1', '2', '3', '4']
# e.g. 'A123B123' -> ['A', '1', '2', '3', 'B', '1', '2', '3']
def split_label(label):
    k_tmp = []
    split_label = []
    for i in label:
        if i.isdigit():
            if len(k_tmp) > 0:
                split_label.append(''.join(k_tmp))
                k_tmp = []
            split_label.append(i)
        else:
            k_tmp.append(i)
    return split_label


# gen label
def gen_label(img_path, label_dict=label_dict):
    img_name = os.path.basename(img_path).replace(' ', '').split('_')[0]
    label = split_label(img_name)
    txt_label = ''.join(label)

    for i, char in enumerate(label):
        label[i] = label_dict[h2j(char)]

    return txt_label, label


# funct to check is double line or not
# give path of txt file, return True if double line, False otherwise
# each line in text file is format as 'x1,y1,x2,y2'
# caculate the center coordinate of each line, if last 4th center coordinate is less than 3st center coordinate, then it is double line
def is_double_line(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    centers = []
    for line in lines:
        x1, y1, x2, y2 = map(int, line.strip().split(' '))
        center = (x1 + x2) / 2
        centers.append(center)

    if centers[2] - centers[-4] < 0:
        return False
    return True


# count double line in a directory
def count_double_line(dir_path):
    single_line_count = 0
    double_line_count = 0
    for f in glob.glob(os.path.join(dir_path, '*.txt')):
        if is_double_line(f):
            double_line_count += 1
        else:
            single_line_count += 1
    print(f"Single line: {single_line_count}")
    print(f"Double line: {double_line_count}")


# counter files in a directory via given extension and directory list
# return total count of files in each directory
def count_files(dirs, exts):
    if not isinstance(dirs, list):
        dirs = [dirs]
    if not isinstance(exts, list):
        exts = [exts]

    counts = []
    for d in dirs:
        count = 0
        for ext in exts:
            count += len(glob.glob(os.path.join(d, f"*.{ext}")))
        counts.append(count)

    # print out the counts
    for d, c in zip(dirs, counts):
        print(f"{d}: {c}")
    return counts


def count_labels_dis():
    r = Romanizer(j2hcj("""가
    나
    다
    라
    마
    거
    너
    더
    러
    머
    버
    서
    어
    저
    고
    노
    도
    로
    모
    보
    소
    오
    조
    구
    누
    두
    루
    무
    부
    수
    우
    주
    하
    허
    호
    바
    사
    아
    자
    배
    서울
    부산
    대구
    인천
    광주
    대전
    울산
    세종
    경기
    강원
    충북
    충남
    전북
    전남
    경북
    경남
    제주"""))
    print(r.romanize())
    _count = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, f"*.{exts[0]}")):
            txt_label, label = gen_label(f)
            for c in label:
                if c not in _count:
                    _count[c] = 0
                _count[c] += 1

    # sort the dict by key
    _count = dict(sorted(_count.items()))
    # print out the counts and replace the key with the label_dict
    for k, v in _count.items():
        char = list(label_dict.keys())[list(label_dict.values()).index(k)]
        # char to utf-8
        r = Romanizer(char)
        roman = r.romanize()
        # print(f"{k}, {char}, {v},")
        print(f"{k}, {char}({roman}), {v},")


# given a dir, draw these images in one image
def draw_images(dir_path, n=9):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i, f in enumerate(glob.glob(os.path.join(dir_path, '*.jpg'))[:n]):
        img = plt.imread(f)
        axs[i // 3, i % 3].imshow(img)
    plt.show()


if __name__ == "__main__":
    dirs = [
        "/Users/haoyu/Documents/datasets/lpr/train",
        "/Users/haoyu/Documents/datasets/lpr/val",
    ]
    exts = ["txt"]
    count_files(dirs, exts)

    # ans = is_double_line("/Users/haoyu/Documents/datasets/lpr/val/서울31사1369_1710221942440003000.txt")
    # ans = is_double_line("/Users/haoyu/Documents/datasets/lpr/val/서울41너8965_1710221972847466000.txt")
    # print(ans)

    # for i in dirs:
    #     count_double_line(i)
    
