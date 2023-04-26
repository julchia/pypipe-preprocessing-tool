from src.core.constants import PREPROCESSING_CONFIG
from src.core.pipeline.pipeline import Pipeline

 
def main() -> None:
    
    pipe_1 = Pipeline(
        pipeline_conf=PREPROCESSING_CONFIG
    ).create_pipeline()
    
    corpus = [
        "HOLA!!1111 gente lindaaaaa!!!",
        "el nombre essssss @Pedro re loco jjajajajjja",
        "mi correo es pedrito@gmail.com",
        "su página es www.pedrito.com ...."
    ]
    
    pipe_1.process_corpus_sequentially(corpus=corpus)
    
    unseen_corpus = [
        "hola gente",
        "jaja su nombre es pedro",
        "el correo de pedro es MAIL"
    ]
    
    vectors = pipe_1.countvec.process(unseen_corpus)

    print(vectors.toarray())


if __name__ == "__main__":
    main()
