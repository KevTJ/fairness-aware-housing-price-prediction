#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/raw"
OUTPUT="${DATA_DIR}/cook_county_data.zip"
DEFAULT_URL="https://raw.githubusercontent.com/KevTJ/kj-project-portfolio/main/Projects/fair-housing-price-prediction/cook_county_data.zip"
DATA_URL="${COOK_COUNTY_DATA_URL:-$DEFAULT_URL}"

mkdir -p "$DATA_DIR"

echo "Downloading Cook County data archive..."
curl --fail --location --retry 3 --retry-delay 2 \
  "$DATA_URL" \
  --output "$OUTPUT"

python - "$OUTPUT" <<'PY'
from pathlib import Path
import sys
import zipfile

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit("Download failed: archive is missing or empty.")

with zipfile.ZipFile(path) as archive:
    names = set(archive.namelist())
    if "cook_county_train.csv" not in names:
        raise SystemExit(
            "Archive downloaded, but cook_county_train.csv was not found inside it."
        )

print(f"Verified archive: {path}")
print(f"Size: {path.stat().st_size / (1024 ** 2):.1f} MiB")
PY

echo "Data are ready at data/raw/cook_county_data.zip"
echo "The data/raw directory is gitignored and will not be committed."
