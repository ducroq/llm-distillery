"""
Quick test to verify the new logging system works correctly.
Tests with just 2 articles to validate all logging components.
"""

from ground_truth.batch_labeler import GenericBatchLabeler

def main():
    print("Testing new logging system...")
    print("="*60)

    # Create labeler instance
    labeler = GenericBatchLabeler(
        prompt_path="prompts/uplifting.md",
        llm_provider="gemini",  # Using Gemini as it's fast and cheap
        output_dir="datasets/test_logging"
    )

    print(f"\nLogging initialized:")
    print(f"- Log file: {labeler.output_dir / 'distillation.log'}")
    print(f"- Metrics file: {labeler.metrics_log_path}")
    print(f"- Error logs: {labeler.output_dir / 'error_logs'}")
    print(f"- Summary: {labeler.output_dir / 'session_summary.json'}")

    # Run with just 2 articles
    print(f"\nProcessing 2 test articles...")
    labeler.run(
        source_file="datasets/raw/master_dataset.jsonl",
        batch_size=2,
        max_batches=1
    )

    print("\n" + "="*60)
    print("Test complete! Check the following files:")
    print(f"1. {labeler.output_dir / 'distillation.log'} - Human-readable log")
    print(f"2. {labeler.metrics_log_path} - Structured metrics (JSONL)")
    print(f"3. {labeler.output_dir / 'session_summary.json'} - Session summary")
    print("="*60)

if __name__ == "__main__":
    main()
