from src.core.constants import PREPROCESSING_CONFIG
from src.core.pipeline import Pipeline


def main() -> None:
    
    pipe_1 = Pipeline(
        pipeline_conf=PREPROCESSING_CONFIG
    ).get_pipe()
    
    corpus = [
        "HOLA!!1111 gente lindaaaaa!!!",
        "el nombre essssss @Pedro re loco jjajajajjja",
        "mi correo es pedrito@gmail.com",
        "su página es www.pedrito.com ...."
    ]
        
    norm_corpus = pipe_1.regex_normalization.normalize_text(corpus)

    print(norm_corpus)
    
    pipe_1.count_vect_featurizer.train(norm_corpus)
    
    unseen_corpus = [
        "hola gente",
        "jaja su nombre es pedro",
        "el correo de pedro es MAIL"
    ]
    
    vectors = pipe_1.count_vect_featurizer.process(unseen_corpus)
    
    print(vectors.toarray())

if __name__ == "__main__":
    main()
