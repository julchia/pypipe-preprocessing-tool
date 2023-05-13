from __future__ import annotations
from typing import List, Dict, Set, Any, Iterable

import os
import pickle
import json
from contextlib import contextmanager


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


def persist_iterable_as_txtfile(data: Iterable, file_dir: str) -> None:
    """
    """
    with open(file_dir, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

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


def check_if_dir_extension_is(to_check: str, dir_path: str) -> bool:
    if dir_path is None:
        return
    extension = os.path.splitext(dir_path)[1].lower()
    if extension in to_check:
        return True
    else:
        return False
  
def lazy_writer(file_path: str, sep: str ="\n") -> None:
    """
    A function that lazily writes strings to a text file.

    It uses the internal function '_lazy_write' to handle file opening and 
    closing. It also uses a yield construct to allow data to bewritten to
    the file lazily.

    The function is initialized using the following code:

        writer = lazy_writer(file_path)
        next(writer)

    and allows strings to be added using a call to sent():

        writer.send(str_obj)

    Args:
        file_path: The path where the text strings will be stored.
        sep: Text separator.
    """
    @contextmanager
    def _lazy_write(file_path=file_path):
        try:
            with open(file_path, "w") as f:
                try:
                    yield f
                finally:
                    f.flush()
        except FileNotFoundError as e:
            raise e
    
    with _lazy_write(file_path) as f:
        while True:
            text = yield
            f.write(text + sep)

