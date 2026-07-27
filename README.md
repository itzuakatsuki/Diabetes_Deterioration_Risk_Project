# Diabetes Deterioration Risk Project

**Patient-grouped clinical risk stratification, CGM Markov/LSTM forecasting, and lifetime medical cost simulation for type 2 diabetes.**

This is a university diabetes research project focused on:

- Patient-grouped clinical risk stratification
- Continuous Glucose Monitoring (CGM) forecasting using Markov chains and LSTM models
- Lifetime medical cost simulation for type 2 diabetes patients

## Repository Structure

```
├── code/                  # Python implementation scripts
├── data/
│   ├── clinical/         # Clinical data files (excluded from git)
│   └── cgm/              # CGM data files (excluded from git)
├── output/
│   ├── clinical/         # Clinical analysis outputs
│   ├── regression/       # Regression model results
│   ├── cgm/              # CGM forecast outputs
│   ├── risk/             # Risk stratification results
│   └── cost/             # Cost simulation outputs
├── docs/                 # Documentation
└── .github/              # GitHub-specific configurations
```

## Data Privacy

This project handles sensitive patient data. Patient-level raw data is never committed to version control. All data files are excluded via `.gitignore`.

## Research Integrity

See `.github/copilot-instructions.md` for research methodology guidelines and best practices.
