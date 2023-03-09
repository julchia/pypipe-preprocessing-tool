from src.core.models.config_models import preprocessing_conf
from src.core.pipeline import Pipeline


def main() -> None:    
    preprocessor_1 = Pipeline(
        pipeline_conf=preprocessing_conf
    ).get_pipeline()
    
    print(type(preprocessor_1))
    
    text = "HOLA!!1111 gente lindaaaaa!!! mi nombre essssss @Pedro re loco jjajajajjja...."
    
    preprocessed_text = preprocessor_1.normalize_text(text=text)
    
    print(preprocessed_text)
    

if name == "main":
    main()
