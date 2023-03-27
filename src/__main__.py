from src.core.constants import PREPROCESSING_CONFIG
from src.core.pipeline import Pipeline


def main() -> None:
    
    preprocessor_1 = Pipeline(
        pipeline_conf=PREPROCESSING_CONFIG
    )
        
    corpus = [
        "HOLA!!1111 gente lindaaaaa!!!",
        "el nombre essssss @Pedro re loco jjajajajjja",
        "mi correo es pedrito@gmail.com",
        "su página es www.pedrito.com ...."
    ]
    
    for sent in corpus:
        print(preprocessor_1.regex_normalization.normalize_text(sent))

if __name__ == "__main__":
    main()
