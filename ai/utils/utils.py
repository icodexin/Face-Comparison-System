from typing import Dict

from prettytable import PrettyTable


def print_config(title: str, config: Dict):
    """打印配置信息字典"""
    print(f'{title}:')
    table = PrettyTable(field_names=['item', 'info'])
    table.add_rows([[k, v] for k, v in config.items()])
    table.align = 'l'
    print(table)
