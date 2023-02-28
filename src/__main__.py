from src.core.constants import NORM_STEPS
from src.core.models.config_models import norm_conf
from src.core.builders.processor_builders import ProcessorBuilderDirector, PreprocessorBuilder


def main() -> None:
    preprocessor_1 = ProcessorBuilderDirector.build(
        builder=PreprocessorBuilder(),
        model_conf=norm_conf,
        steps_to_build=NORM_STEPS,
    )
    
    text = "HOLA!!1111 mi nombre es Pedro...."
    
    preprocessed_text = preprocessor_1.preprocess_text(text=text)
    
    print(preprocessed_text)
    

if __name__ == "__main__":
    main()
