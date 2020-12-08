import os
from torch.utils.data import Dataset
from strong_transform import augmentation, trans
import json
import random
import cv2


class DfdcDataset(Dataset):

    def __init__(self, datapath="", phase='train', resize=(320, 320)):
        assert phase in ['train', 'val', 'test']
        if phase == 'val':
            phase = 'valid'
        self.phase = phase
        self.resize = resize
        self.num_classes = 2
        self.epoch = 0
        self.next_epoch()
        self.aug = augmentation
        self.trans = trans
        self.datapath = datapath

    def next_epoch(self):
        with open('part_dfdc.json') as f:
            dfdc = json.load(f)
        # self.dataset=dfdc['train']
        if self.phase == 'train':
            trainset = dfdc['train']#+dfdc['valid']
            tr = list(filter(lambda x: x[1] == 0, trainset))  # 选取其中标号为0的视频，真实视频
            # tf = random.sample(list(filter(lambda x: x[1] == 1, trainset)), len(tr))  # 根据真实视频的数量选取假视频
            tf = list(filter(lambda x: x[1] == 1, trainset))  # 选取其中标号为0的视频，真实视频
            self.dataset = tr+tf  # 合并成数据集
        if self.phase == 'valid':
            validset = dfdc['test']   # 从test中选取数据
            tr = list(filter(lambda x: x[1] == 0, validset))
            # tf = random.sample(list(filter(lambda x: x[1] == 1, validset)), len(tr))
            tf = list(filter(lambda x: x[1] == 1, validset))  # 选取其中标号为0的视频，真实视频
            self.dataset = tr+tf
        if self.phase == 'test':
            self.dataset = dfdc['test']
        self.epoch += 1

    def __getitem__(self, item):
        try:
            vid = self.dataset[item//20]
            # print("info | vid:",vid)
            ind = str(item % 20*12+self.epoch % 12)
            # ind = '0'*(3-len(ind))+ind+'.png'
            ind = '0'*(3-len(ind))+ind+'.jpg'
            image = cv2.imread(os.path.join(self.datapath, vid[0], ind))
            # if not image:
            #     ind = str(random.randint(80))
            #     ind = '0'*(3-len(ind))+ind+'.jpg'
            #     image = cv2.imread(os.path.join(self.datapath, vid[0], ind))
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.resize)
            if self.phase == 'train':
                image = self.aug(image=image)['image']
            return self.trans(image), vid[1]
        except:
            # cou = input()
            return self.__getitem__((random.randint(0,len(self.dataset)*20)+random.randint(0,80) )% (self.__len__()))

    def __len__(self):
        return len(self.dataset)*20
