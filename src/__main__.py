from src.core.models.config_models import preprocessing_pipeline
from src.core.builders.process_builders import ProcessorBuilderDirector


def main() -> None:
    preprocessor_1 = ProcessorBuilderDirector(
        pipeline=preprocessing_pipeline
    ).build_preprocessor()
    
    text = "HOLA!!1111 gente lindaaaaa!!! mi nombre essssss @Pedro re loco jjajajajjja...."
    
    preprocessed_text = preprocessor_1.normalize_text(text=text)
    
    print(preprocessed_text)
    

if __name__ == "__main__":
    main()
