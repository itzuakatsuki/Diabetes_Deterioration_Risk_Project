# Diabetes Deterioration Risk Project — Research Integrity Guidelines

These instructions ensure scientific rigor and ethical compliance throughout the project.

## Core Principles

1. **Never fabricate research results**
   - All findings must be derived from actual data analysis
   - Do not invent metrics, statistics, or conclusions

2. **Never commit patient-level raw data**
   - All patient identifiers and raw clinical data are excluded from version control
   - Use `.gitignore` to prevent accidental uploads
   - Only committed aggregated or processed outputs are acceptable

3. **Use patient-grouped validation for repeated clinical records**
   - Apply stratification strategies at the patient level, not individual visit level
   - Validate splits to ensure records from the same patient don't leak between train/test sets

## Statistical and Modeling Requirements

4. **Keep preprocessing inside sklearn Pipeline**
   - Ensures consistent preprocessing during training and inference
   - Prevents data leakage from manual preprocessing steps
   - Facilitates reproducibility and auditing

5. **Use out-of-fold predictions for evaluation**
   - Never report training-set metrics as test metrics
   - Apply cross-validation strictly: models trained on fold A are evaluated on fold B only
   - Test set must remain completely separate from model development

6. **Never report training-set metrics as test metrics**
   - Clearly distinguish between training, validation, and test performance
   - Report final results only from held-out test sets
   - Document which metrics come from which data partition

## Time-Series and Forecasting Requirements

7. **Do not create Markov transitions or LSTM windows across invalid timestamp gaps**
   - Check for missing time steps, measurement gaps, or data quality issues
   - Only create transitions/windows within continuous, valid time ranges
   - Document any gaps or exclusions applied to the time series

## Causal Inference

8. **Do not describe cross-sectional associations as causal effects**
   - Acknowledge observational data limitations
   - Use language like "associated with" rather than "causes"
   - Never imply causality without appropriate study design or sensitivity analysis

## Repository Practices

9. **Keep only one root README.md**
   - Main project documentation lives at the repository root
   - Subdirectories should not contain their own README files
   - Use `.github/copilot-instructions.md` for methodology guidelines

10. **Keep random_state=42 unless explicitly documented otherwise**
    - Use a fixed random seed for reproducibility
    - If random_state must differ (e.g., for sensitivity analysis), document the reason
    - Ensure all team members use consistent seeds

## Prohibited Actions

- Do not create placeholder Python scripts
- Do not commit model results, CSV files, figures, or datasets to the repository root
- Do not commit medical conclusions or research findings as code
- Do not bypass .gitignore rules

## Questions or Updates?

If these guidelines need refinement, open an issue or pull request with proposed changes.
