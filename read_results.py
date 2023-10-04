import os
import numpy as np
import pandas as pd

def convert_percentage_to_decimal(percentage_str):
    # 移除百分号
    decimal_str = percentage_str.replace('%', '')
    # 将字符串转换为浮点数
    decimal_float = float(decimal_str)
    # 转换为小数
    decimal_float /= 100
    return decimal_float


def list_last_level_files(directory_path):
    # 初始化文件列表和目录深度
    files_at_last_level = []
    deepest_level = 11

    # 使用os.walk遍历目录树
    for dirpath, dirnames, filenames in os.walk(directory_path):
        current_level = dirpath.count(os.sep)

        if current_level == deepest_level:
            files_at_last_level.extend([os.path.join(dirpath, filename) for filename in filenames])

    return files_at_last_level

directory_path = './output/'
df = pd.DataFrame(columns=['Experiment Name', 'Result'])
file_list = list_last_level_files(directory_path)
data = []
for file in file_list:
    try:
        if '4shots' in file and 'rn50' in file:
            if 'baseline' in file:
                _tmp = file.split('/')
                dataset_name = _tmp[2]
                method = _tmp[3]
                model_name = _tmp[4]

                with open(file) as f:
                    lines = f.readlines()
                    acc = convert_percentage_to_decimal(lines[-13].strip('\n').strip('-').strip(']').split(', ')[-1])
                    data.append({'Experiment Name': dataset_name + '_' + method + '_' + model_name, 'Result': acc})
            else:
                _tmp = file.split('/')
                dataset_name = _tmp[2]
                method = _tmp[3]
                model_name = _tmp[4]

                with open(file) as f:
                    lines = f.readlines()
                    acc = convert_percentage_to_decimal(lines[-7].strip('\n').split(': ')[1])
                    data.append({'Experiment Name': dataset_name + '_' + method + '_' + model_name, 'Result': acc})
    except IndexError:
        print(file)
        pass

df = pd.DataFrame(data) * 100

mean = df.groupby('Experiment Name')['Result'].mean()
variance = df.groupby('Experiment Name')['Result'].var() / 100

df_stats = pd.DataFrame({'Mean': mean, 'Variance': variance})

print(df_stats)
