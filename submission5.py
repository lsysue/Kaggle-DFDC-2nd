# 原模型结构 + 概率计算trick + 模型集成trick
import os
import sys
import time

import cv2
import numpy as np
from collections import defaultdict

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from face_utils import norm_crop, FaceDetector
from model_def import WSDAN, xception


class DFDCLoader:
    def __init__(self, video_dir, face_detector, transform=None,
                 batch_size=25, frame_skip=9, face_limit=25):
        self.video_dir = video_dir
        self.file_list = sorted(f for f in os.listdir(video_dir))# if f.endswith(".mp4"))

        self.transform = transform
        self.face_detector = face_detector

        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.face_limit = face_limit

        self.record = defaultdict(list)
        self.score = defaultdict(lambda: 0.5)
        # 推断时间
        self.infer_start = defaultdict(lambda: 0)
        self.infer_end = defaultdict(lambda: 0)
        self.feedback_queue = []

    def iter_one_face(self):
        for fname in self.file_list:
            path = os.path.join(self.video_dir, fname)
            reader = cv2.VideoCapture(path)
            face_count = 0

            while True:
                for _ in range(self.frame_skip):
                    reader.grab()

                success, img = reader.read()
                if not success:
                    break

                boxes, landms = self.face_detector.detect(img)
                if boxes.shape[0] == 0:
                    continue

                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()

                boxes = boxes[order]
                landms = landms[order]

                # Crop faces
                landmarks = landms.numpy().reshape(5, 2).astype(np.int)
                img = norm_crop(img, landmarks, image_size=320)
                aligned = Image.fromarray(img[:, :, ::-1])

                if self.transform:
                    aligned = self.transform(aligned)

                yield fname, aligned

                # Early stop
                face_count += 1
                if face_count == self.face_limit:
                    break

            reader.release()

    def __iter__(self):
        self.record.clear()
        self.feedback_queue.clear()

        batch_buf = []
        t0 = time.time()
        batch_count = 0
        last_name = ""
        for fname, face in self.iter_one_face():
            if last_name != fname and last_name != "":
                yield torch.stack(batch_buf)

                batch_count += 1
                batch_buf.clear()
                if batch_count % 10 == 0:
                    elapsed = 1000 * (time.time() - t0)
                    print("T: %.2f ms / batch" % (elapsed / batch_count))
                    # 测试控制
                    # break
            if fname != last_name:
                self.feedback_queue.append(fname)
            if not batch_buf:   # batch_buf is clear
                self.infer_start[fname] = time.time()
            batch_buf.append(face)
            last_name = fname

        if len(batch_buf) > 0:
            yield torch.stack(batch_buf)

    def feedback(self, pred):
        fname = self.feedback_queue.pop(0)
        self.score[fname] = pred
        print(pred)
        self.infer_end[fname] = time.time()
        print("[%s] %.6f | %.3f ms" % (fname, self.score[fname], (self.infer_end[fname] - self.infer_start[fname]) * 1000))

def main(arg_dir):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    test_dir = arg_dir

    face_detector = FaceDetector()
    face_detector.load_checkpoint("./pretrained_weights/RetinaFace-Resnet50-fixed.pth")
    loader = DFDCLoader(test_dir, face_detector, T.ToTensor())

    model1 = xception(num_classes=2, pretrained=False)
    # ckpt = torch.load("./pretrained_weights/xception-hg-2.pth")
    ckpt = torch.load("./trained_weights/dfdc-xception.pth/")
    model1.load_state_dict(ckpt["state_dict"])
    model1 = model1.cuda()
    model1.eval()

    model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).cuda()
    # ckpt = torch.load("./pretrained_weights/ckpt_x.pth")
    ckpt = torch.load("./trained_weights/dfdc-wsdan-x.pth/")
    model2.load_state_dict(ckpt["state_dict"])
    model2.eval()

    model3 = WSDAN(num_classes=2, M=8, net="efficientnet", pretrained=False).cuda()
    # ckpt = torch.load("./pretrained_weights/ckpt_e.pth")
    ckpt = torch.load("./trained_weights/dfdc-wsdan-e.pth/")
    model3.load_state_dict(ckpt["state_dict"])
    model3.eval()

    zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).cuda()
    zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).cuda()

    for batch in loader:
        batch = batch.cuda(non_blocking=True)
        i1 = F.interpolate(batch, size=299, mode="bilinear")
        i1.sub_(0.5).mul_(2.0)
        o1 = model1(i1).softmax(-1)[:, 1].cpu().numpy()

        i2 = (batch - zhq_nm_avg) / zhq_nm_std
        o2, _, _ = model2(i2)
        o2 = o2.softmax(-1)[:, 1].cpu().numpy()

        i3 = F.interpolate(i2, size=300, mode="bilinear")
        o3, _, _ = model3(i3)
        o3 = o3.softmax(-1)[:, 1].cpu().numpy()

        track_probs = np.concatenate((o1, o2, o3))

        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())
        loader.feedback(video_score)

    with open("result5.csv", "w") as fout:
        for fname in loader.file_list:
            pred = loader.score[fname]
            start_time = loader.infer_start[fname] * 1000
            end_time = loader.infer_end[fname] * 1000
            print("%s\t%.6f\t%d\t%d" % (fname, pred, start_time, end_time), file=fout)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("ERR | too less arguments given. (2 expected)")
    arg_dir = sys.argv[1]
    main(arg_dir)
