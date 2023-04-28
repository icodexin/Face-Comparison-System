import numpy as np
import cv2
import base64


def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def base64_to_bytes(base64_str: str) -> bytes:
    return base64.b64decode(base64_str)


def bytes_to_base64(bytes_obj: bytes) -> str:
    return base64.b64encode(bytes_obj).decode('utf-8')


def list_to_dict(lst):
    """
    Converts a list to a dictionary with list index as key and list item as value.
    """
    dictionary = {}
    for i, item in enumerate(lst):
        dictionary[i] = item
    return dictionary


def dict_to_list(dictionary):
    """
    Converts a dictionary to a list of tuples.
    """
    lst = [value for key, value in dictionary.items()]
    return lst
