import logging
import logging.handlers
from config import log_dirpath


def logger() -> logging.Logger:
    # 获得应用的日志器
    return logging.getLogger('FCS')


# 设置基本日志级别
logger().setLevel(logging.DEBUG)

# 日志文件
log_file_handler = logging.handlers.TimedRotatingFileHandler(
    filename=f"{log_dirpath}/fcs.log",
    when='midnight',
    backupCount=7,
    encoding='utf-8'
)
log_file_handler.suffix = '%Y-%m-%d.log'

# 终端
terminal_handler = logging.StreamHandler()

# 日志格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s')

# 设置各handler的日志级别和格式
for handler in [log_file_handler, terminal_handler]:
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

logger().addHandler(terminal_handler)
logger().addHandler(log_file_handler)
