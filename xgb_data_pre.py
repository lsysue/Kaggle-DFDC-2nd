import os
import sys
import time
import csv
import simplejson as json

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
        self.file_list = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))

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
            # print("fname",fname)
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
            self.feedback_queue.append(fname)

            if not batch_buf:   # batch_buf is clear
                self.infer_start[fname] = time.time()
            batch_buf.append(face)
            last_name = fname

        if len(batch_buf) > 0:
            yield torch.stack(batch_buf)

    def file_for_xgboost(self, o1, o2, o3, label):
        with open("./model_scores.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([o1, o2, o3, label])

def main(arg_dir):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    test_dir = arg_dir

    face_detector = FaceDetector()
    face_detector.load_checkpoint("./pretrained_weights/RetinaFace-Resnet50-fixed.pth")
    loader = DFDCLoader(test_dir, face_detector, T.ToTensor())

    model1 = xception(num_classes=2, pretrained=False)
    model1.load_state_dict(ckpt["state_dict"])
    model1 = model1.cuda()
    model1.eval()

    model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).cuda()
    ckpt = torch.load("./pretrained_weights/ckpt_x.pth")
    model2.load_state_dict(ckpt["state_dict"])
    model2.eval()

    model3 = WSDAN(num_classes=2, M=8, net="efficientnet", pretrained=False).cuda()
    ckpt = torch.load("./pretrained_weights/ckpt_e.pth")
    model3.load_state_dict(ckpt["state_dict"])
    model3.eval()

    zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).cuda()
    zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).cuda()

    # load label message
    # 将json文件的groundtruth信息读到一个列表中
    data_list = []
    with open('./part_dfdc.json', 'r') as f:
        json_data = json.load(f)    #此时json_data是一个字典对象
        for key in json_data:  # 遍历字典a获取key
            key_data_list = json_data[key]
            for k_data in key_data_list:
                data_list.append(k_data)

    # 将列表转换为字典，方便读取groundtruth
    data_dict = {}
    for data in data_list:
        data_dict[data[0]] = data[1]
    
    with open("./model_scores1.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model1_score", "model2_score", "model3_score", "ground_truth"])

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

        for i in range(loader.batch_size):
            video_name = loader.feedback_queue.pop(0).split('.')[0]
        label = data_dict[video_name]
        # print("video_name: ", video_name)

        batch_size = o1.size
        for i in range(0,batch_size):
            loader.file_for_xgboost(o1[i], o2[i], o3[i], label)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("ERR | too less arguments given. (2 expected)")
    arg_dir = sys.argv[1]
    main(arg_dir)
