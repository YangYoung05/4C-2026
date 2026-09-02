# Visualization data snapshot

This directory contains the minimal, derived data required to rebuild the web
visualization without committing the full raw and cleaned datasets.

- The CSV and JSON files are finalized outputs from the Python analysis.
- `china-provinces.geojson` is the geometry snapshot consumed by the web app.
- Run `npm run data` to regenerate `public/data` from this directory.
- Set `THUNDER_PROJECT_ROOT`, `THUNDER_ASSET_ROOT`, `THUNDER_CLEAN_ROOT`, or
  `THUNDER_EXTERNAL_DATA_ROOT` to rebuild from a full local analysis workspace.

The snapshot supports visualization reproducibility. Reproducing the complete
statistical analysis still requires the external datasets listed in
`雷霆医疗队/08_data_inventory/数据目录索引.md`.
