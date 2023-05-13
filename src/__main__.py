from src.core import paths
from src.core.pipeline.pipeline import Pipeline


def main() -> None:
    
    corpus = [
        "HOLA!!1111 gente lindaaaaa!!!",
        "el nombre essssss @Pedro re loco jjajajajjja",
        "mi correo es pedrito@gmail.com",
        "su página es www.pedrito.com ...."
    ]
    
    pipe_1 = Pipeline(
        corpus=corpus,
        config_path=paths.PREPROCESSING_CONFIG_PATH
    )
        
    pipe_1.run_processes_sequentially(persist=True)
    
    unseen_corpus = [
        "hola gente",
        "jaja su nombre es pedro",
        "el correo de pedro es MAIL"
    ]
    
    countvec = pipe_1.create_pipeline_process(alias="countvec")
    
    countvec.load()
    
    vectors = countvec.process(corpus=unseen_corpus)

    print(vectors.toarray())


if __name__ == "__main__":
    main()

