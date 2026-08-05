# Gemini Extraction Pipeline: Input Data Format

The pipeline accepts raw clinical datasets in CSV, Excel (`.xlsx`, `.xls`), or Parquet (`.parquet`) formats.

---

## Dataset Format Requirements

1. **Text Column** (Default: `input_text`):
   - Contains the raw clinical text (e.g., discharge summary, consultation report, imaging narrative).
   - Configurable in `config.yaml` under `data.column_mapping.input_text`.

2. **Patient Identifier Column** (Default: `patient_id`):
   - Unique record identifier.
   - Configurable in `config.yaml` under `data.column_mapping.patient_id`.

3. **Optional Ground Truth Columns**:
   - Ground truth fields (e.g., `ground_truth_mRS`, `ground_truth_smoking_status`, `ground_truth_aneurysm_size_mm`), if present in the input file, are preserved in the output result CSV for evaluation.

---

## Synthetic Dataset Generation

Generate a sample clinical dataset for local testing:

```bash
python scripts/generate_synthetic_data.py
```

This creates `data/synthetic_clinical_notes.csv` containing 7 sample clinical notes in French and English with ground truth annotations.