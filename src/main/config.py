import sys
sys.path.append('..')

# 可以通过更改选择config_e2e里包含的不同参数组合
from configs.config_e2e.config_main import *

class hyperparams:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

# Instantiate the class with the config_dict
args = hyperparams(config_dict)
# print(f"当前运行程序参数：{args}")