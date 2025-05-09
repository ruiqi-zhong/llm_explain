import re
from typing import Union
from llm_explain import logger

def extract_tag_from_output(output: str, tag: str) -> Union[str, None]:
    try:
        return re.search(rf"<{tag}>(.*?)</{tag}>", output).group(1)
    except Exception as e:
        logger.warning(f"Error extracting tag {tag} from output {output}: {e}")
        return None
    
    