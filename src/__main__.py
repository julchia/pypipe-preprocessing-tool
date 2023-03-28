from src.core.constants import PREPROCESSING_CONFIG
from src.core.pipeline import Pipeline


# CORREGIR TODOS LOS NOMBRES DE LOS ARCHIVOS Y REORDENARLOS
# VER SI EL VECTORIZADOR PUEDE VECTORIZAR DESDE UN STR
# AGRENA DESDE LA PIPELINE UNA FORMA DE EJECUTAR TODOS LOS PROCESS

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
    
    normalized_corpus = []
    for sent in corpus:
        norm_sent = pipe_1.regex_normalization.normalize_text(sent)
        normalized_corpus.append(norm_sent)
        
    pipe_1.featurization.train(normalized_corpus)

if __name__ == "__main__":
    main()
