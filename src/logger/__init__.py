import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)

log_file_path = os.path.join(log_dir_path, LOG_FILE)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
)

# File Handler
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)