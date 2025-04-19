# utils/logger.py

import logging
import os
from datetime import datetime

# Create a logs directory
os.makedirs("logs", exist_ok=True)

# Log filename with timestamp (can also use fixed name like "pipeline.log")
log_file = os.path.join("logs", "project_log.log")

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
