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

# Mammographic Mass Classification with a Decision Tree

## Project Overview

This project focuses on building a **Machine Learning classification model using a Decision Tree Classifier** to predict whether a mammographic mass is benign or malignant.

The project uses the Mammographic Mass Dataset, which contains patient and mammographic characteristics. The goal is to explore the dataset, perform data preprocessing, and develop a classification model to understand how machine learning can support mammographic mass classification.

This project is developed as part of my Data Science and Machine Learning learning journey.

## Project Objectives

- Explore and understand the Mammographic Mass Dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify and handle missing values.
- Prepare features for machine learning.
- Train a Decision Tree Classification model.
- Evaluate model performance using classification metrics.
- Interpret the model's predictions and results.

## Dataset

**Dataset:** Mammographic Mass Dataset

**Source:** Kaggle

[Download Mammographic Mass Dataset](https://www.kaggle.com/datasets/overratedgman/mammographic-mass-data-set?resource=download)

The dataset originates from the UCI Machine Learning Repository.

### Dataset Features

| Feature | Description |
|---|---|
| BI-RADS | BI-RADS assessment |
| Age | Patient's age |
| Shape | Shape of the mammographic mass |
| Margin | Margin characteristics |
| Density | Density of the mass |
| Severity | Target variable (Benign / Malignant) |

**Dataset characteristics:**
- Total instances: 961
- Input features: 5
- Target variable: Severity
- Classification type: Binary Classification
- Target classes: Benign and Malignant

## Technologies and Libraries

The following tools and libraries are used in this project:

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

The project follows these major steps:

1. **Data Loading:** Import the dataset using Pandas.
2. **Data Cleaning:** Identify and handle missing values.
3. **Exploratory Data Analysis:** Examine feature distributions and relationships.
4. **Data Preprocessing:** Prepare the dataset for model training.
5. **Train-Test Split:** Divide the dataset into training and testing sets.
6. **Model Building:** Implement a Decision Tree Classifier.
7. **Model Evaluation:** Evaluate the model using appropriate classification metrics.
8. **Result Interpretation:** Analyze the model's performance and limitations.

## Machine Learning Model

**Algorithm:** Decision Tree Classifier

Decision Trees are supervised machine learning algorithms used for classification and regression tasks.

In this project, the algorithm learns patterns from the training data to classify mammographic masses into two categories:

- Benign
- Malignant

## Model Evaluation

The model is evaluated using classification performance metrics, including:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

These metrics help assess how effectively the model distinguishes between benign and malignant cases.

## Repository Structure

```text
Mammographic_Mass_Classification_with_a_Decision_Tree/
│
├── main.md
|    └──README.md
├──experiment.md
     └── mammographic_masses_classification_with_a_Decision_Tree.ipynb
     └── mammographic_masses_clean.csv
```

*Note: The structure above is illustrative. Actual filenames and folders may differ.*

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree.git
```

2. Navigate to the project directory:

```bash
cd Mammographic_Mass_Classification_with_a_Decision_Tree
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `experiment.ipynb` and execute the notebook cells.

Alternatively, you can run the notebook using Google Colab.

## Important Disclaimer

This project is developed for educational and research purposes only. The model is not intended for clinical diagnosis or medical decision-making. Its predictions should not replace professional medical evaluation.

## Author

**Jahura Begum**

Aspiring Data Analyst | Data Science and Machine Learning Enthusiast

- GitHub: [Jahura7](https://github.com/Jahura7)
- Project Repository: [Mammographic Mass Classification with a Decision Tree](https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree)
- Dataset: [Mammographic Mass Dataset – Kaggle](https://www.kaggle.com/datasets/overratedgman/mammographic-mass-data-set?resource=download)

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback!
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

### 3. Run the notebook

Run the notebook cells from top to bottom using **Runtime → Run all** (or run each cell individually).

The notebook uses common Python data-science libraries: Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. If Colab reports that a required package is missing, install it in a code cell:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Review the outputs

After execution, review the EDA visualizations, baseline comparison, tuned Decision Tree results, classification report, feature importance, and decision-tree visualization.

## Limitations

- The cleaned dataset used in the notebook contains 830 records and may not represent all patient populations or clinical settings.
- The notebook uses numeric target labels (0 and 1); their mapping to class names should be confirmed from the original dataset documentation.
- A single random train/test split does not establish generalization across hospitals, time periods, or different populations.
- The feature-importance values and decision-tree rules do not demonstrate causation.
- The reported metrics are experimental results, not evidence of clinical effectiveness.
- This project is for learning and portfolio demonstration only. It must not be used for diagnosis, screening decisions, or treatment decisions.

## Possible Future Improvements

- Compare the Decision Tree with Logistic Regression and Random Forest using the same evaluation strategy.
- Use repeated or nested cross-validation to assess model-selection stability.
- Evaluate probability calibration and class-specific error costs.
- Confirm and clearly document the target-label mapping.
- Validate the model on an independent dataset before making any claims about generalization.

## Author

**Jahura Begum**

Data Science and Machine Learning learner

- [GitHub Profile](https://github.com/Jahura7)
- [Project Repository](https://github.com/Jahura7/Mammographic_Mass_Classification_with_a_Decision_Tree)

---

If you explore this project, feedback and suggestions are welcome.
