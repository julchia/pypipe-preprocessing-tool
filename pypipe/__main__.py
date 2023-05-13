import sys
import argparse

from pypipe.core import paths
from pypipe.core.pipeline.pipeline import Pipeline


def main():
    """
    Pypipe Pipeline CLI.

    Parses command line arguments, loads configuration file, and runs the 
    specified or sequentially pipeline processes.

    Usage:
    python -m pypipe <config_alias or config_path> [--corpus <path_to_corpus>] [--store] 
                [--process <alias_of_pipeline_process> --method <method_to_execute>] 

    args:
        <config_alias or config_path>: Alias of configuration file or path to 
            configuration file.
            
        --corpus <path_to_corpus>: Path to corpus file.
        
        --store: Persist output of each process.
        
        --process <alias_of_pipeline_process>: Alias of the pipeline process 
            to run.
            
        --method <method_to_execute>: Method to execute for the specified 
            pipeline process.

    Returns:
        None
    """
    config_alias = {
        "preprocessing_1": paths.PREPROCESSING_CONFIG_PATH
    }
    
    parser = argparse.ArgumentParser(description="Pipeline CLI")
    parser.add_argument("config", help="Alias of configuration file or path to configuration file")
    parser.add_argument("--corpus", "-c", help="Path to corpus file")
    parser.add_argument("--store", "-s", action="store_true", help="Persist output of each process")
    parser.add_argument("--process", "-p", help="Alias of the pipeline process to run")
    parser.add_argument("--method", "-m", required="--process" in sys.argv, help="Method to execute for the specified pipeline process")
    args = parser.parse_args()
    
    if args.config in config_alias:
        config_path = config_alias[args.config]
    else:
        config_path = args.config
    
    pipeline = Pipeline(config_path, corpus=args.corpus)
    
    if args.process:
        process = pipeline.create_pipeline_process(args.process)
        getattr(process, args.method)(corpus=pipeline._corpus, persist=args.store)
    else:
        pipeline.run_processes_sequentially(persist=args.store)


if __name__ == "__main__":
    main()

