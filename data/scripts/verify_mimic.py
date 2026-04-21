"""
Verify that the required MIMIC-III CSV files are present in the data/mimic/ directory.

Run via:
    python data/scripts/verify_mimic.py
or
    make verify-mimic
"""

import os
import sys
from pathlib import Path

MIMIC_PATH = os.getenv("MIMIC_DATA_PATH", str(Path(__file__).parent.parent / "mimic"))

REQUIRED_FILES = {
    "PATIENTS.csv":    "Core demographics (DOB, gender, DOD)",
    "ADMISSIONS.csv":  "Hospital admissions with diagnosis codes",
    "ICUSTAYS.csv":    "ICU stay durations and unit info",
    "CHARTEVENTS.csv": "Vital signs and numeric chart observations",
    "LABEVENTS.csv":   "Lab results (creatinine, bilirubin, platelets, etc.)",
    "NOTEEVENTS.csv":  "Clinical free-text notes (discharge summaries, nursing notes)",
}

OPTIONAL_FILES = {
    "D_ITEMS.csv":     "Data dictionary for CHARTEVENTS ITEMIDs",
    "D_LABITEMS.csv":  "Data dictionary for LABEVENTS ITEMIDs",
}


def main():
    print(f"\nChecking MIMIC-III files at: {MIMIC_PATH}\n")

    missing = []
    for fname, description in REQUIRED_FILES.items():
        path = Path(MIMIC_PATH) / fname
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓  {fname:<25} {size_mb:>8.1f} MB  —  {description}")
        else:
            print(f"  ✗  {fname:<25} MISSING         —  {description}")
            missing.append(fname)

    print()
    for fname, description in OPTIONAL_FILES.items():
        path = Path(MIMIC_PATH) / fname
        status = "✓" if path.exists() else "–"
        print(f"  {status}  {fname:<25} (optional)     —  {description}")

    print()
    if missing:
        print(f"ERROR: {len(missing)} required file(s) missing.")
        print("  → Download from https://physionet.org/content/mimiciii/1.4/")
        print(f"  → Place them in: {MIMIC_PATH}/")
        sys.exit(1)
    else:
        print("All required MIMIC-III files present. Ready to run the pipeline.")


if __name__ == "__main__":
    main()
