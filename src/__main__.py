from src.core.constants import PREPROCESSING_CONFIG
from src.core.pipeline import Pipeline


def main() -> None:    
    preprocessor_1 = Pipeline(
        pipeline_conf=PREPROCESSING_CONFIG
    ).get_pipeline()
        
    text = "HOLA!!1111 gente lindaaaaa!!! mi nombre essssss @Pedro re loco jjajajajjja y mi correo es pedrito@gmail.com y mi página es www.pedrito.com ...."
    
    preprocessed_text = preprocessor_1.normalize_text(text=text)
    
    print(preprocessed_text)
    

if __name__ == "__main__":
    main()
