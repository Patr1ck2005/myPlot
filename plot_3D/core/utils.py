# utils.py
import re
def safe_str(val):
    return re.sub(r'[^\w.-]', '', str(val))
