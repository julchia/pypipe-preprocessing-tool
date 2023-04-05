from src.core.constants import PREPROCESSING_CONFIG
from src.core.pipeline import Pipeline


def main() -> None:
    
    pipe_1 = Pipeline(
        pipeline_conf=PREPROCESSING_CONFIG
    )
    
    corpus = [
        "HOLA!!1111 gente lindaaaaa!!!",
        "el nombre essssss @Pedro re loco jjajajajjja",
        "mi correo es pedrito@gmail.com",
        "su página es www.pedrito.com ...."
    ]

    norm_corpus = pipe_1.regex_normalization.normalize_text(corpus)
        
    pipe_1.sklearn_count_vect.train(norm_corpus)

if __name__ == "__main__":
    main()
