import xgboost as xgb
import pandas as pd
from xgboost import plot_importance
# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#读入数据
train = pd.read_csv("model_scores.csv")

#用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
train_xy,val = train_test_split(train, test_size = 0.3,random_state=1)

y = train_xy.ground_truth
X = train_xy.drop(["ground_truth"],axis=1)
val_y = val.ground_truth
val_X = val.drop(["ground_truth"],axis=1)

#xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
# xgb_test = xgb.DMatrix(tests)
 
# 算法参数
params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'gamma':0.1,
    'max_depth':5,
    'lambda':3,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'slient':1,
    'eta':0.1,
    'seed':1000,
    'nthread':4,
    'scale_pos_weight':0.25,
    'verbosity':0
}
num_rounds = 300
plst = list(params.items())
model = xgb.train(plst,xgb_train,num_rounds)
 
# 对测试集进行预测
ans = model.predict(xgb_val)
print(ans)
 
# 显示重要特征
# plot_importance(model)
# plt.show()

model.save_model('./model_def/xgb.model') # 用于存储训练出的模型
