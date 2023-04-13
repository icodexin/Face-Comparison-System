from typing import Dict

from prettytable import PrettyTable
from utils.logger import logger


def print_config(title: str, config: Dict):
    """打印配置信息字典"""
    print_str = f'{title}:\n'
    table = PrettyTable(field_names=['item', 'info'])
    table.add_rows([[k, v] for k, v in config.items()])
    table.align = 'l'
    print_str += table.get_string()
    logger().info(print_str)
