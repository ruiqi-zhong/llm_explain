import re
import logging

logging_level = logging.DEBUG

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


def extract_tag_from_text(text: str, tag: str) -> str | None:
    try:
        return re.search(rf"<{tag}>(.*?)</{tag}>", text).group(1)
    except Exception as e:
        logger.warning(f"Error extracting tag {tag} from text {text}: {e}")
        return None
    
    