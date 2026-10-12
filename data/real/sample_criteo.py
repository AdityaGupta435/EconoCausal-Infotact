
import csv
import gzip
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "real" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
OUTPUT_DIR = ROOT / "data" / "real" / "processed"
OUTPUT = OUTPUT_DIR / "criteo_sample.csv"

SAMPLE_SIZE = 100_000
SEED = 42


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Reservoir sampling: keeps memory usage bounded.
    reservoir = []
    header = None
    total_rows = 0

    with gzip.open(SOURCE, "rt", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            if header is None:
                header = row
                continue

            total_rows += 1

            if len(reservoir) < SAMPLE_SIZE:
                reservoir.append(row)
            else:
                index = rng.randrange(total_rows)
                if index < SAMPLE_SIZE:
                    reservoir[index] = row

            if total_rows % 1_000_000 == 0:
                print(f"Rows scanned: {total_rows:,}")

    if not header or not reservoir:
        raise ValueError("No data rows were found.")

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(reservoir)

    print("\nSampling complete.")
    print(f"Total source rows scanned: {total_rows:,}")
    print(f"Sample rows written: {len(reservoir):,}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
