import itertools
import argparse
import glob
import os


parser = argparse.ArgumentParser(description="Generate LSTM hyperparameter yaml files.")
parser.add_argument("--config-dir", type=str, default="./Data/hyperpara")
args = parser.parse_args()

config_dir = args.config_dir
os.makedirs(config_dir, exist_ok=True)

pattern = f"{config_dir}/*.yaml"  # 也可以是 "C:/path/to/files/*.log", "**/*.bak" 等
files_to_delete = glob.glob(pattern)

for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"✅ 已删除: {file_path}")
    except Exception as e:
        print(f"❌ 删除失败 {file_path}: {e}")


def write_yaml_config(path, params):
    with open(path, 'w') as file_e:
        for key, value in params.items():
            file_e.write(f"{key}: {value}\n")

# Original step2-style grid kept for reference:
# param_grid = {
#         "epoch_begin":           [0],
#         "epochs":                [5000],
#         "learning_rate":         [0.0001],
#         "neurons":               [128, 64],
#         "l2":                    [0.001, 0.002],
#         "dropout":               [0.05, 0.08],
#         "batch_size":            [64],
#         "look_back":             [365],
#         }
#
# Previous sun-only weighted grid kept for reference:
# param_grid = {
#         "epoch_begin":           [0],
#         "epochs":                [5000],
#         "learning_rate":         [0.0001],
#         "neurons":               [64, 32],
#         "l2":                    [0.003, 0.005],
#         "dropout":               [0.10, 0.20],
#         "batch_size":            [64],
#         "look_back":             [365],
#         }
#
# Sun-only weighted grid: lower capacity and stronger regularization.
param_grid = {
        "epoch_begin":           [0],
        "epochs":                [5000],
        "learning_rate":         [0.0001],
        "neurons":               [32, 16],
        "l2":                    [0.005, 0.01],
        "dropout":               [0.20, 0.30],
        "batch_size":            [64],
        "look_back":             [365],
        }

# 数据集划分比例要作为成对方案处理，避免 train_num 和 val_num 做笛卡尔积。
split_grid = [
        {"train_num": 0.6, "val_num": 0.2},
        {"train_num": 0.7, "val_num": 0.15},
        ]

# 生成所有组合
param_combinations = list(itertools.product(*param_grid.values()))

# 为每个组合创建配置文件
config_id = 0
for split_params in split_grid:
    for combination in param_combinations:
        params = dict(zip(param_grid.keys(), combination))
        params.update(split_params)
        write_yaml_config(f"{config_dir}/paras_{config_id}.yaml", params)
        config_id += 1

print(f"Generated {config_id} yaml files in {config_dir}")
