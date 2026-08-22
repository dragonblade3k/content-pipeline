"""
cli.py

Usage:
    python3 cli.py --topic senna
    python3 cli.py --list-topics
"""

import argparse
import os
from pathlib import Path

from pipeline.pipeline import Pipeline
from pipeline.research import StaticF1FactSource


def main():
    parser = argparse.ArgumentParser(description="Generate a short form F1 facts clip.")
    parser.add_argument("--topic", help="Topic to generate a clip for, e.g. senna")
    parser.add_argument("--list-topics", action="store_true", help="List available topics and exit")
    parser.add_argument("--output-dir", default="outputs", help="Where to write the finished clip")
    args = parser.parse_args()

    if args.list_topics or not args.topic:
        if os.environ.get("PIPELINE_FACTS", "static") == "live":
            print("PIPELINE_FACTS=live uses open ended '<driverId>-<season>' topics, "
                  "e.g. verstappen-2024, there's no fixed list to print.")
        else:
            print("Available topics:")
            for t in StaticF1FactSource().topics():
                print(f"  - {t}")
        if not args.topic:
            return

    pipeline = Pipeline.from_env(output_dir=Path(args.output_dir))
    result = pipeline.run(args.topic)

    print(f"\nTitle:    {result.metadata.title}")
    print(f"Caption:  {result.metadata.caption_with_hashtags()}")
    print(f"Video:    {result.video_path}")


if __name__ == "__main__":
    main()
