import logging

logging_level = logging.INFO

# Define the logger
logger = logging.getLogger(__name__)

logger.setLevel(logging_level)

# Define the format string
format_str = (
    "===%(asctime)s - %(filename)s - Line %(lineno)d - %(levelname)s ===\n %(message)s\n\n"
)
date_format = "%Y-%m-%d %H:%M:%S"  # Define a date format

# Create a console handler and set its level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging_level)
console_handler.setFormatter(logging.Formatter(format_str, datefmt=date_format))

logger.addHandler(console_handler)
logger.debug("Logger initialized")

import llm_explain.models as models
import llm_explain.llm as llm
import llm_explain.utils as utils

__all__ = ["models", "llm", "utils"]
