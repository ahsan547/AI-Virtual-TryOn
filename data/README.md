# Dataset Directory Structure

This directory contains all datasets and trained models for the virtual try-on recommendation system.

## Directory Structure

```
data/
├── raw/              # Place all downloaded Kaggle dataset ZIP files here
├── processed/        # Extracted and cleaned datasets (auto-generated)
└── models/           # Trained ML models (auto-generated)
```

## Instructions

### 1. Downloading Datasets
- Download all Kaggle datasets as ZIP files
- Place them directly in the `data/raw/` directory
- Keep the original ZIP filenames (or use descriptive names)

### Example:
```
data/raw/
├── face-shape-classification.zip
├── eyewear-recommendation-dataset.zip
├── glasses-catalog-attributes.zip
├── face-attributes-dataset.zip
└── user-preferences-fashion.zip
```

### 2. Dataset Types to Download

Priority datasets to look for:
1. **Face Shape Classification** - Images with labeled face shapes
2. **Eyewear/Glasses Recommendation** - User preferences and ratings
3. **Glasses Product Catalog** - Detailed attributes (shape, color, style, etc.)
4. **Face Attributes** - Skin tone, facial features classification
5. **Fashion Preferences** - User interaction data (views, likes, purchases)

### 3. Next Steps

After placing the ZIP files:
- The ML pipeline will automatically extract and process them
- Models will be trained and saved in `data/models/`
- Processed datasets will be stored in `data/processed/`

## Notes

- Don't modify files in `data/processed/` or `data/models/` manually
- Keep original ZIP files in `data/raw/` for reference
- Each dataset will be processed according to its type and purpose

