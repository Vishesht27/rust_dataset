import pandas as pd
from collections import Counter
import re

import argparse

parser = argparse.ArgumentParser(description="Error analysis on the validated Rust dataset.")
parser.add_argument("--filepath", type=str, help="Input file path", required=True)
args = parser.parse_args()

# Load CSV
df = pd.read_csv(args.filepath)

# Extract error codes or main error messages
def extract_error(msg):
    if pd.isna(msg):
        return None
    # Match Rust error codes like error[E0405]
    match = re.search(r"error\[[A-Z0-9]+\]", msg)
    if match:
        return match.group(0)
    # Fallback: take first part of the error message
    return msg.split("-->")[0].strip()

# Apply extraction
df["error_type"] = df["stdout"].apply(extract_error)

# Count frequency
error_counts = Counter(df["error_type"].dropna())

# Get top N errors
N = 5
top_errors = error_counts.most_common(N)

# Print results
print(f"Top {N} most common errors:")
for err, count in top_errors:
    print(f"{err}: {count}")
