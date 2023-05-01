from typing import List, Dict, Type
import tempfile
import json
import pickle


def create_temp_pickle_file(
    obj: Type, 
    suffix: str, 
    delete: bool = False
) -> None:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=delete) as f:
        pickle.dump(obj, f)
        return f.name
     
     
def create_temp_json_file_from_dict(
    obj: Dict, 
    mode: str ="w", 
    delete: bool = False
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=delete, mode=mode) as f:
        json.dump(obj, f)
        return f.name
    
    
def create_temp_txt_file_from_list(
    obj: List[str], 
    mode: str = "w", 
    delete: bool = False
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=delete, mode=mode) as f:
        f.write("\n".join(obj))
        return f.name
    