"""
CLI Script to generate, validate, and split the synthetic telehealth dataset.
Usage:
    python scripts/generate_dataset.py [--num_conversations 4000] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# Safe stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import generate_and_save_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic telehealth symptom dataset.")
    parser.add_argument("--num_conversations", type=int, default=4000,
                        help="Number of multi-turn conversations to generate (~2 records per conversation).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print("[INFO] Starting synthetic dataset generation...")
    print(f"       Target conversations: {args.num_conversations} (~{args.num_conversations * 2} records)")
    print(f"       Random seed: {args.seed}")

    manifest = generate_and_save_dataset(num_conversations=args.num_conversations, seed=args.seed)

    print("\n[SUCCESS] Dataset Generation & Validation Complete!")
    print(f"       Total Statements : {manifest['total_statements']}")
    print(f"       Total Mentions   : {manifest['total_mentions']}")
    print(f"       Splits           : Train={manifest['split_counts']['train']}, "
          f"Val={manifest['split_counts']['val']}, Test={manifest['split_counts']['test']}")
    print("       Category Distribution (Train):")
    for cat, count in manifest['train_category_distribution'].items():
        print(f"         - {cat:20s}: {count}")

if __name__ == "__main__":
    main()
