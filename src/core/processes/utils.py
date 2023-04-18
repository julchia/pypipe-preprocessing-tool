from __future__ import annotations
from typing import List, Dict, Set, Any
import os
import pickle
import json


def create_dir_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def open_line_by_line_txt_file(
    file_dir: str, 
    mode: str = "r", 
    as_set: bool = False  
) -> List | Set:
    """
    """
    if file_dir.endswith('.txt'):
        with open(file_dir, mode) as f:
            txt = [line.strip("\n") for line in f]
            if as_set:
                return set(txt)
            return txt


def open_json_as_dict(file_dir: str) -> Dict[str, Any]:
    """
    """
    if file_dir.endswith('.json'):
        with open(file_dir) as f:
            return json.load(f)


def persist_dict_as_json(
    file: Dict[str, Any], 
    mode: str,
    file_dir: str,
) -> None:
    """
    """
    with open(file_dir, mode) as f:
        json.dump(file, f, indent=4)
        

def persist_data_with_pickle(
    file_to_persist: Any,  
    mode: str,
    file_dir: str
) -> None:
    """
    """
    with open(file_dir, mode) as f:
        pickle.dump(file_to_persist, f)
        

def load_data_with_pickle(
    mode: str,
    file_dir: str
) -> None:
    """
    """
    return pickle.load(open(file_dir, mode))
    
    