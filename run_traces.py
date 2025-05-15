from agent_eval.trace_reader import TraceReader
import config as config

def main():
    """
    Main function to run the trace reader.
    """
    # Create TraceReader instance with config values
    reader = TraceReader(
        project_name=config.PROJECT_NAME,
        span_name=config.SPAN_NAME,
        trace_id=config.TRACE_ID,
        output_dir=config.OUTPUT_DIR,
        output_filename_template=config.OUTPUT_FILENAME_TEMPLATE
    )
    
    # Process trace and save to file
    try:
        reader.process_trace()
        print(f"Successfully processed trace.")
    except Exception as e:
        print(f"Failed to process trace: {e}")

if __name__ == "__main__":
    main()