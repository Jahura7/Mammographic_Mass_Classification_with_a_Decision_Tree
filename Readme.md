# Mammographic Mass Classification with a Decision Tree

## Project Overview

This project applies a **Decision Tree classifier** to a mammographic mass dataset to predict a binary target label from five input features: BI-RADS assessment, age, shape, margin, and density.

The notebook covers data validation, exploratory data analysis (EDA), a stratified train/test split, a Dummy Classifier baseline, hyperparameter tuning with cross-validation, final test-set evaluation, feature importance, and decision-tree visualization.

This is an educational machine-learning portfolio project. It is **not a medical diagnostic tool**.

## Project Links

- **GitHub repository:** [Mammographic Mass Classification with a Decision Tree](https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree)
- **Experiment notebook:** [Mammographic_Mass_Classification_with_a_Decision_Tree.ipynb](https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree/blob/experiment.md/Mammographic_Mass_Classification_with_a_Decision_Tree.ipynb) 
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

## Tools and Libraries

- **Google Colab** — notebook development and execution environment
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

The analysis was developed and executed in **Google Colab**. A local Jupyter installation is not required to reproduce the notebook.

## Methodology

1. **Load and validate data:** Read the cleaned CSV, inspect its shape, data types, and missing values, and validate the binary target.
2. **Exploratory data analysis:** Review target counts, age distributions, and feature correlations.
3. **Prepare features and target:** Use `BI-RADS`, `Age`, `Shape`, `Margin`, and `Density` as predictors; use `Target` as the target variable.
4. **Split the data:** Create a stratified 80/20 train/test split with `random_state=42` (664 training rows and 166 test rows).
5. **Establish a baseline:** Evaluate a most-frequent Dummy Classifier.
6. **Build a pipeline:** Apply median imputation followed by a Decision Tree classifier. The imputer is kept inside the pipeline to avoid fitting it on validation folds. The current cleaned dataset has no missing values.
7. **Tune the model:** Use `GridSearchCV` with 5-fold stratified cross-validation and F1 scoring. Search over tree criterion, maximum depth, minimum samples per split/leaf, and class weighting.
8. **Evaluate and interpret:** Evaluate the selected model on the held-out test set, inspect feature importance, and visualize the selected tree.

Feature scaling is not used because it is not required for a Decision Tree.

## Results

### Final test-set performance

The following results are taken from the executed notebook's held-out test set (166 records):

| Metric | Test score |
|---|---:|
| Accuracy | 0.8554 |
| Balanced accuracy | 0.8542 |
| Precision (label 1) | 0.8904 |
| Recall (label 1) | 0.8025 |
| F1-score (label 1) | 0.8442 |
| ROC-AUC | 0.8726 |

### Classification report

| Target label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.83 | 0.91 | 0.87 | 85 |
| 1 | 0.89 | 0.80 | 0.84 | 81 |

### Cross-validation

The selected model achieved a mean cross-validation F1-score of **0.8185** for label 1.

Selected hyperparameters:

| Parameter | Selected value |
|---|---|
| `class_weight` | `balanced` |
| `criterion` | `entropy` |
| `max_depth` | `3` |
| `min_samples_leaf` | `1` |
| `min_samples_split` | `2` |

### Baseline comparison

| Model | Accuracy | Balanced accuracy | F1-score (label 1) |
|---|---:|---:|---:|
| Most-frequent Dummy Classifier | 0.5120 | 0.5000 | 0.0000 |
| Tuned Decision Tree | 0.8554 | 0.8542 | 0.8442 |

The tuned Decision Tree outperformed the most-frequent baseline on the reported test metrics.

### Feature importance

| Feature | Importance |
|---|---:|
| BI-RADS | 0.7173 |
| Shape | 0.1759 |
| Age | 0.0785 |
| Margin | 0.0143 |
| Density | 0.0139 |

In this fitted tree, BI-RADS had the highest feature-importance value, followed by Shape and Age. Feature importance describes the model's use of features; it does not establish causation.

## How to Run (Google Colab)

This project was developed in Google Colab. Follow these steps to run the notebook in the cloud.

### 1. Open the notebook in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Open notebook → GitHub**.
3. Paste the project repository URL:

   `https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree`

4. Select the `experiment` branch and open the project notebook, `Mammographic_Mass_Classification_with_a_Decision_Tree.ipynb`.

You can also download the `.ipynb` file from the repository and upload it to Colab using **File → Upload notebook**.

### 2. Upload the cleaned dataset

The notebook reads a CSV file using this filename:

```text
mammographic_masses_data_clean.csv
```

Before running the data-loading cell:

1. Download or prepare the cleaned CSV file.
2. In Colab, open the **Files** panel on the left.
3. Click **Upload** and select `mammographic_masses_data_clean.csv`.
4. Confirm that the file appears in the Colab session's working directory.

The CSV must contain the columns `BI-RADS`, `Age`, `Shape`, `Margin`, `Density`, and `Target`. The notebook's data-loading cell uses the relative path `mammographic_masses_data_clean.csv`.

> **Note:** Files uploaded directly to a Colab session are temporary and may need to be uploaded again when the runtime is restarted. For persistent storage, you can keep the CSV in Google Drive and mount Drive in Colab, then update the notebook's CSV path to the file's Drive location.
