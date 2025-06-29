# BMI-EYE-model
Some other codes coming soon...

# Deep Survival Analysis for Ophthalmic Disease Risk Prediction

This repository contains the source code for the paper titled "[Your Paper Title Here]". It implements and compares several survival analysis models, from traditional statistical methods to deep learning architectures, for predicting the long-term risk of ophthalmic diseases.

## Models Implemented

This project provides implementations for the following five survival analysis models:

1.  **DeepSurv**: A deep learning extension of the Cox Proportional Hazards model.
2.  **DeepHit**: A discrete-time deep learning model that handles competing risks and non-proportional hazards.
3.  **Survival Transformer**: A novel survival model utilizing a Transformer architecture to capture complex feature interactions.
4.  **Cox Proportional Hazards (CoxPH)**: The classic semi-parametric statistical model for survival analysis (wrapper for `lifelines`).
5.  **Random Forest Survival**: An ensemble-based, non-parametric survival model.

Each model is encapsulated in its own Python module for clarity and reusability.

## Directory Structure

```
.
├── deepsurv_model.py
├── deephit_model.py
├── transformer_model.py
├── cox_model.py
├── random_forest_model.py      
└── README.md
```

## Requirements

The project is developed using Python 3.8+. We recommend creating a virtual environment to manage dependencies.

Key dependencies include:

  * PyTorch
  * pandas
  * scikit-learn
  * lifelines
  * matplotlib
  * seaborn

## Data

**Please note:** The data used in this study is sourced from the **UK Biobank**.

In accordance with the UK Biobank's data usage agreement, **we are not permitted to publicly release or share the raw dataset**.

Researchers who wish to access this data must submit an application directly through the official UK Biobank website:
[**https://www.ukbiobank.ac.uk/**](https://www.ukbiobank.ac.uk/)

## Citation

If you use the code from this repository in your research, please consider citing our paper:

```
[Waiting]
```
