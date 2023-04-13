import os
from pprint import pprint
from typing import Dict
import yaml

# 打开配置文件
config_path = 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 获取配置信息
fastapi_config: Dict = config['fastapi']['config']
fastapi_log_fmt: str = config['fastapi']['log_format']
log_dirpath: str = config['log_dirpath']
if not os.path.exists(log_dirpath):
    os.mkdir(log_dirpath)
server_config: Dict = config['server']