from face_utils import FaceDetector, norm_crop

import os
import sys
import itertools

import cv2
import torch
import numpy as np
import random
from PIL import Image


def main():
    torch.set_grad_enabled(False)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    reader = cv2.VideoCapture(input_file)
    face_detector = FaceDetector()
    face_detector.load_checkpoint("./pretrained_weights/RetinaFace-Resnet50-fixed.pth")

    for idx in itertools.count():
        success, img = reader.read()
        if not success:
            break

        boxes, landms = face_detector.detect(img)
        if boxes.shape[0] == 0:
            continue

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        max_face_idx = areas.argmax()
        landm = landms[max_face_idx]

        landmarks = landm.numpy().reshape(5, 2).astype(np.int)
        img = norm_crop(img, landmarks, image_size=320)

        # 进行下采样并进行上采样 
        down_size = random.randint(1, 10)  # 10表示最多图片缩小为原来的十倍
        # print(down_size)
        if down_size != 1:
            dim_down = (round(320/down_size), round(320/down_size))
            img = cv2.resize(img, dim_down, interpolation=cv2.INTER_AREA)
            # print(img_downsampled.shape)
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
            # print(img_upsampled.shape)

        aligned = Image.fromarray(img[:, :, ::-1])

        out_path = os.path.join(output_dir, "%03d.jpg" % idx)
        aligned.save(out_path)


if __name__ == "__main__":
    main()
