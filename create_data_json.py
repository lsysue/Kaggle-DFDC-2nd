import json
import os

data_json= {"train": [], "valid": []}
data_list = []
with open('./dfdc.json', 'r') as f:
    json_data = json.load(f)    #此时json_data是一个字典对象
    # for key in json_data:  # 遍历字典a获取key
    #     print(key)
    train_data_list = json_data["train"]
    # print(train_data_list)
    for t_data in train_data_list:
        data_list.append(t_data)
    valid_data_list = json_data["valid"]
    for v_data in valid_data_list:
        data_list.append(v_data)
    # print(valid_data_list)
    test_data_list = json_data["test"]
    for test_data in test_data_list:
        data_list.append(test_data)
    testall_data_list = json_data["test_all"]
    for testall_data in testall_data_list:
        data_list.append(testall_data)
# print(data_list)
data_dict = {}
for data in data_list:
    # print(data)
    data_dict[data[0]] = data[1]
# print(data_dict)

data_path_list = ['./data/train_sample_videos', './data/dfdc_train_part_45', './data/dfdc_train_part_46']

for data_path in data_path_list:
    data_name = os.listdir(data_path)
    # print(data_name)
    data_num = len(data_name)
    current_num = 0
    for name in data_name:
        # print(name)
        temp = name.split('.')
        if temp[1] == 'mp4':
            label = data_dict[temp[0]]
            # print(label)
            if current_num <= (data_num*0.7):
                data_json["train"].append([name, label])
            else:
                data_json["valid"].append([name, label])
            current_num += 1

# print(data_json)
with open('./part_dfdc.json', 'w') as f:
    json.dump(data_json,f)

