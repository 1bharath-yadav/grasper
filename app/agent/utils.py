import numpy as np
from typing import Any, Dict, List, Union


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy data types to native Python types for JSON serialization.

    Args:
        obj: The object to convert

    Returns:
        The object with all numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, str):
        return obj
    elif obj is None:
        return obj
    elif isinstance(obj, (int, float, bool)):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        try:
            # Check if it's a numpy type we haven't handled
            if hasattr(obj, 'dtype'):
                if np.issubdtype(obj.dtype, np.integer):
                    return int(obj)
                elif np.issubdtype(obj.dtype, np.floating):
                    return float(obj)
                elif np.issubdtype(obj.dtype, np.bool_):
                    return bool(obj)
            return str(obj)
        except:
            return str(obj)
