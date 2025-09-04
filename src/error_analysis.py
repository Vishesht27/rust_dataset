import pandas as pd
from collections import Counter
import itertools
import re

import argparse

parser = argparse.ArgumentParser(description="Error analysis on the validated Rust dataset.")
parser.add_argument("--filepath", type=str, help="Input file path", required=True)
parser.add_argument("--top_n", type=int, default=5, help="Number of top errors to display")
args = parser.parse_args()

# Load CSV
df = pd.read_csv(args.filepath)

# Extract error codes or main error messages
def extract_errors(msg):
    if pd.isna(msg):
        return None
    
    errors = []

    # 1. Capture all Rust error codes like error[E0432]
    codes = re.findall(r"error\[[A-Z0-9]+\]", msg)
    errors.extend(codes)

    # 2. Capture general error messages without codes
    # Matches lines starting with "error:" but not followed by [E...]
    generic_errors = re.findall(r"(?m)^error:(?!\[)[^\n]+", msg)
    for e in generic_errors:
        e = e.strip()
        # Exclude "could not compile ..." summary lines
        if not re.match(r"error: could not compile .+ due to \d+ previous errors?", e):
            errors.append(e)


    # Deduplicate while preserving order
    seen = set()
    unique_errors = []
    for e in errors:
        if e not in seen:
            unique_errors.append(e)
            seen.add(e)

    return unique_errors if unique_errors else None

# Apply extraction
df["error_type"] = df["stdout"].apply(extract_errors)

# Count frequency
all_errors = list(itertools.chain.from_iterable(df["error_type"].dropna()))
error_counts = Counter(all_errors)

# Get top N errors
N = args.top_n
top_errors = error_counts.most_common(N)

# Print results
print(f"Top {N} most common errors:")
for err, count in top_errors:
    print(f"{err}: {count}")
