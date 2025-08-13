import pandas as pd

# Read the CSV (try auto-detect separator just in case)
df = pd.read_csv("homographies_batch4_v4.csv", sep=None, engine="python")

# Save it back with proper Excel-friendly formatting
df.to_csv("homographies_batch4_v4_fix.csv", index=False, sep=",", quoting=1)  # 1 = csv.QUOTE_NONNUMERIC

