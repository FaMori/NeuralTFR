"""
Preprocess TFR CSV data into a JS file for the web visualization.
Uses data/estimates/empirical_tfr.csv for raw historical points.
Uses data/estimates/empirical_mean.csv for historical means.
Uses data/forecast/NeuralTFR.csv and data/forecast/WPP.csv for forecasts.
Only includes countries present in the NeuralTFR forecast dataset.
Outputs data.js with a global variable so it works with file:// protocol.
"""
import csv
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FORECAST_DIR = os.path.join(DATA_DIR, "forecast")
ESTIMATES_DIR = os.path.join(DATA_DIR, "estimates")

EMPIRICAL_TFR_FILE = os.path.join(ESTIMATES_DIR, "empirical_tfr.csv")
EMPIRICAL_MEAN_FILE = os.path.join(ESTIMATES_DIR, "empirical_mean.csv")
NEURAL_FORECAST_FILE = os.path.join(FORECAST_DIR, "NeuralTFR.csv")
WPP_FORECAST_FILE = os.path.join(FORECAST_DIR, "WPP.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data.js")

# --- Helpers ---

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def safe_float(val):
    if val is None or val == "":
        return None
    try:
        return round(float(val), 4)
    except ValueError:
         return None

# --- Step 1: Determine the set of valid country IDs from NeuralTFR ---

neural_rows = read_csv(NEURAL_FORECAST_FILE)
valid_ids = set(int(row["id"]) for row in neural_rows)
print(f"NeuralTFR countries: {len(valid_ids)}")

# --- Step 2: Build country id -> name mapping ---
# Names come from empirical_mean.csv; for countries not in that file
# (shouldn't happen after augmentation, but kept as a safety net),
# fall back to empirical_tfr.csv.

id_to_name = {}
for path in (EMPIRICAL_MEAN_FILE, EMPIRICAL_TFR_FILE):
    for row in read_csv(path):
        cid = int(row["id"])
        if cid in valid_ids and cid not in id_to_name:
            id_to_name[cid] = row["name"]

countries = sorted(
    [{"id": cid, "name": name} for cid, name in id_to_name.items()],
    key=lambda c: c["name"],
)
print(f"Countries with names: {len(countries)}")

# --- Step 3: Historical TFR Means ---
print(f"Reading empirical mean data from {EMPIRICAL_MEAN_FILE}...")
mean_rows = read_csv(EMPIRICAL_MEAN_FILE)

historical_mean = defaultdict(list)
for row in mean_rows:
    cid = int(row["id"])
    if cid not in valid_ids:
        continue
    tfr = safe_float(row.get("TFR"))
    if tfr is not None:
        year = int(row["year"])
        historical_mean[cid].append({
            "year": year,
            "tfr": tfr
        })

for cid in historical_mean:
    historical_mean[cid].sort(key=lambda x: x["year"])

print(f"Historical means: {len(historical_mean)} countries")

# --- Step 4: Historical TFR Points ---
print(f"Reading empirical points data from {EMPIRICAL_TFR_FILE}...")
points_rows = read_csv(EMPIRICAL_TFR_FILE)

historical_points = defaultdict(list)
for row in points_rows:
    cid = int(row["id"])
    if cid not in valid_ids:
        continue
    tfr = safe_float(row.get("TFR"))
    if tfr is not None:
        year = int(row["year"])
        # Handle differing column names
        source = row.get("source", "")
        method = row.get("method", "")
        historical_points[cid].append({
            "year": year,
            "tfr": tfr,
            "source": source,
            "method": method
        })

for cid in historical_points:
    historical_points[cid].sort(key=lambda x: x["year"])

print(f"Historical points: {len(historical_points)} countries")


# --- Step 5: Forecast predictions ---

FORECAST_FILES = {
    "NeuralTFR": NEURAL_FORECAST_FILE,
    "WPP": WPP_FORECAST_FILE,
}

forecast_models = {}
for model_name, path in FORECAST_FILES.items():
    rows = read_csv(path)
    model_data = defaultdict(list)
    for row in rows:
        cid = int(row["id"])
        if cid not in valid_ids:
            continue
        entry = {
            "year": int(row["year"]),
            "median": safe_float(row.get("y_hat_50")),
            "lower": safe_float(row.get("y_hat_low")),
            "upper": safe_float(row.get("y_hat_upp")),
        }
        model_data[cid].append(entry)
    for cid in model_data:
        model_data[cid].sort(key=lambda x: x["year"])
    forecast_models[model_name] = dict(model_data)

# --- Step 6: Compute forecast year ranges per model ---

model_year_ranges = {}
for model_name, model_data in forecast_models.items():
    all_years = set()
    for cid_data in model_data.values():
        for d in cid_data:
            all_years.add(d["year"])
    if all_years:
        model_year_ranges[model_name] = {
            "min": min(all_years),
            "max": max(all_years),
        }

print("Model forecast year ranges:")
for m, yr in model_year_ranges.items():
    print(f"  {m}: {yr['min']}-{yr['max']}")

# --- Step 7: Write JSON ---

output = {
    "countries": countries,
    "historical_mean": dict(historical_mean),
    "historical_points": dict(historical_points),
    "forecast": forecast_models,
    "modelYearRanges": model_year_ranges,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("window.FERTCAST_DATA = ")
    json.dump(output, f, ensure_ascii=False)
    f.write(";\n")

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"Wrote {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
print(f"Models (forecast): {list(forecast_models.keys())}")
