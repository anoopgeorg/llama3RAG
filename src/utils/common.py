import yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
import os


def read_yaml(path: Path) -> ConfigBox:
    """
        Reads YAML file and return ConfigBox object
    Args:
        path:Path - Path of YAML file
    Return:
        ConfigBox - ConfigBox object
    """
    try:
        with open(path, "r") as file:
            content = yaml.safe_load(file)
            print(f"YAML file loaded successfully from: {path}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def create_directory(dir_list: list, verbose=True):
    """
        Creates the directories from a list
    Args:
        dir_list:list - list with paths of directories to be created
    """
    for path in dir_list:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory {path}")
