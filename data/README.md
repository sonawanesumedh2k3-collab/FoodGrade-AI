# FoodGrade AI V2 data

- `raw/` stores raw Open Food Facts API responses after an import.
- `processed/` stores cleaned FoodGrade CSV files.
- `processed/foodgrade_india_latest.csv` is the latest cleaned export.

The V2 import is intentionally small (10–100 products per action). We are building the dataset pipeline first; we are not bulk-downloading the entire Open Food Facts database.
