import sys
import math
from collections import defaultdict

def cross_entropy(gtruth_csv, result_csv):
    with open(gtruth_csv) as gf, open(result_csv) as rf:
        next(gf)
        num = 0  # 计数预测的个数
        cross_entropy_loss = 0.0
        gtruth = defaultdict(lambda:0.0)
        result = defaultdict(lambda:0.0)
        # 处理result_csv
        while True:
            rline = rf.readline()
            if not rline:
                break
            fname = rline.split(".")[0]
            if 0 == int(rline.split("\t")[2]):
                continue
            result[fname] = float(rline.split("\t")[1]) # 读取预测结果
            num = num + 1

        while True:
            gline = gf.readline()
            if not gline:
                break
            fname = gline.split(",")[0]
            if fname in result.keys():
                gtruth[fname] = float(gline.split(",")[1]) # 读取实际结果
                cross_entropy_loss = cross_entropy_loss - (gtruth[fname] * math.log2(result[fname]) + (1 - gtruth[fname]) * math.log2(1 - result[fname]))
        cross_entropy_loss = cross_entropy_loss / num
    return cross_entropy_loss, num

if __name__=="__main__":
    if len(sys.argv) <= 2:
        print("ERR | too less arguments given. (3 expected")
    loss, video_num = cross_entropy(sys.argv[1], sys.argv[2])
    print("RESULT | cross entropy loss: %.8f | video num: %d" % (loss, video_num))
