import itertools
import yaml

import glob
import os

pattern = "./YAML/*.yaml"  # 也可以是 "C:/path/to/files/*.log", "**/*.bak" 等
files_to_delete = glob.glob(pattern)

for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"✅ 已删除: {file_path}")
    except Exception as e:
        print(f"❌ 删除失败 {file_path}: {e}")

# 定义参数空间
param_grid = {
        "epoch_begin":           [0],
        "epochs":                [5000],
        "learning_rate":         [0.0001],
        "neurons":               [128, 64],
        "l2":                    [0.001, 0.002],
        "dropout":               [0.05, 0.08],
        "batch_size":            [64 ],
        "train_num":             [0.6],
        "val_num":               [0.2],
        "look_back":             [365],
        }

# 生成所有组合
param_combinations = list(itertools.product(*param_grid.values()))

# 为每个组合创建配置文件
for i, combination in enumerate(param_combinations):
    params = dict(zip(param_grid.keys(), combination))
    with open(f"YAML/paras_{i}.yaml", 'w') as file_e:
        yaml.dump(params, file_e)
