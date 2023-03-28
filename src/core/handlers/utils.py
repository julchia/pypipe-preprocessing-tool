from __future__ import annotations
from typing import List, Dict, Set, Any
import os
import pickle
import json
from re import sub


class SubRegexBuilder(str):
    """
    It extends the str class to implement a builder pattern 
    that allows the sub function to be applied multiple 
    times.
    """

    def __new__(cls, *args, **kwargs):
        newobj = str.__new__(cls, *args, **kwargs)
        newobj.sub = lambda fro, to: SubRegexBuilder(sub(fro, to, newobj))
        return newobj


def create_dir_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def open_line_by_line_txt_file(
    path: str, 
    mode: str = "r", 
    as_set: bool = False
    
) -> List | Set:
    """
    """
    if path.endswith('.txt'):
        with open(path, mode) as f:
            txt = [line.strip("\n") for line in f]
            if as_set:
                return set(txt)
            return txt


def open_json_as_dict(path: str) -> Dict[str, Any]:
    """
    """
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)


def persist_dict_as_json(
    file: Dict[str, Any],
    path: str, 
    mode: str = "w",
    indent: int = 4
) -> None:
    """
    """
    with open(path, mode) as f:
        json.dump(file, f, indent=indent)
        

def persist_data_with_pickle(
    file_to_persist: str, 
    file_dir: str, 
    mode: str
) -> None:
    """
    """
    with open(file_dir, mode) as f:
        pickle.dump(file_to_persist, f)
        

def load_data_with_pickle(
    path: str,
    mode: str = "rb"
) -> None:
    """
    """
    return pickle.load(open(path, mode))
    
    