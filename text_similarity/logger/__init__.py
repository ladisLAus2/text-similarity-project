import logging
import os

from from_root import from_root
from datetime import datetime

logging_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", logging_file)

os.makedirs(logs_path, exist_ok=True)

logging_file_path = os.path.join(logs_path, logging_file)

logging.basicConfig(
    filename=logging_file_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)