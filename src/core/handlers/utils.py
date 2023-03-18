import os
import pickle
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
    

def persist_data_with_pickle(file_to_persist: str, file_dir: str, mode: str) -> None:
    with open(file_dir, mode) as f:
        pickle.dump(file_to_persist, f)
        

def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)