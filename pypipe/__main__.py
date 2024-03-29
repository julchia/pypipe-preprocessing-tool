import sys
import argparse

from pypipe.core.pipeline.pipeline import Pipeline


def main() -> None:
    """
    Pypipe Pipeline CLI.

    Parses command line arguments, loads configuration file, and runs the 
    specified or sequentially pipeline processes.

    Usage:
    python -m pypipe <config_alias or config_path> [--data <path_to_data>] [--store] 
                [--process <alias_of_pipeline_process> --method <method_to_execute>] 

    args:
        <config_alias or config_path>: Alias of configuration file or path to 
            configuration file.
            
        --data <path_to_data>: Path to data corpus file.
        
        --store: Persist output of each process.
        
        --process <alias_of_pipeline_process>: Alias of the pipeline process 
            to run.
            
        --method <method_to_execute>: Method to execute for the specified 
            pipeline process.

    Returns:
        None
    """    
    parser = argparse.ArgumentParser(description="Pipeline CLI")
    parser.add_argument("config",help="Alias of configuration file or path to configuration file")
    parser.add_argument("--data", "-d", default=None, help="Path to data file")
    parser.add_argument("--store", "-s", action="store_true", help="Persist output of each process")
    parser.add_argument("--process", "-p", help="Alias of the pipeline process to run")
    parser.add_argument("--method", "-m", required="--process" in sys.argv, help="Method to execute for the specified pipeline process")
    args = parser.parse_args()
        
    pipeline = Pipeline(args.config, data=args.data)
    
    if args.process:
        process = pipeline.create_pipeline_process(args.process)
        try:
            getattr(process, args.method)(data=args.data, persist=args.store)
        except TypeError:
            getattr(process, args.method)(data=args.data)
    else:
        pipeline.run_processes_sequentially(persist=args.store)


if __name__ == "__main__":
    main()

