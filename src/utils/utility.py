from datetime import datetime

def get_filename(*prefixes, suffix=".json"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "_".join(prefixes)
    return f"{prefix_str}_{timestamp}{suffix}"
