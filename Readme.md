# Mammographic Mass Classification with a Decision Tree

## Project Overview

This project applies a **Decision Tree classifier** to a mammographic mass dataset to predict a binary target label from five input features: BI-RADS assessment, age, shape, margin, and density.

The notebook covers data validation, exploratory data analysis (EDA), a stratified train/test split, a Dummy Classifier baseline, hyperparameter tuning with cross-validation, final test-set evaluation, feature importance, and decision-tree visualization.

This is an educational machine-learning portfolio project. It is **not a medical diagnostic tool**.

## Project Links

- **GitHub repository:** [Mammographic Mass Classification with a Decision Tree](https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree)
- **Experiment notebook:** [Mammographic_Mass_Classification_with_a_Decision_Tree.ipynb](https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree/blob/experiment/Mammographic_Mass_Classification_with_a_Decision_Tree.ipynb) 
- **Execution environment:** [Google Colab](https://colab.research.google.com/)
- **Dataset:** [Mammographic Mass Dataset on Kaggle](https://www.kaggle.com/datasets/overratedgman/mammographic-mass-data-set?resource=download)
## Dataset

The notebook loads a cleaned CSV file named `mammographic_masses_data_clean.csv`.

The executed notebook contains **830 records and 6 columns**:

| Column | Role |
|---|---|
| `BI-RADS` | Input feature |
| `Age` | Input feature |
| `Shape` | Input feature |
| `Margin` | Input feature |
| `Density` | Input feature |
| `Target` | Binary target label (0 or 1) |

The notebook reports 427 records with target label `0` and 403 with target label `1`. It checks that the target contains exactly the two labels, `0` and `1`.

> **Target-label note:** The notebook uses numeric labels 0 and 1 but does not document their class-to-name mapping in the executed cells. Confirm the mapping against the dataset documentation and record it before interpreting class-specific results as benign or malignant.
