# DA5401-JUL-NOV-2025-assignment-4-shriprasad15

# Project Mangatha: GMM-Based Synthetic Sampling for Imbalanced Data

**Course:** DA5401 - Advanced Data Analytics
**Assignment:** A4 - GMM-Based Synthetic Sampling for Imbalanced Data

---

## 1. Project Objective

This project tackles the critical challenge of class imbalance in machine learning, specifically within the context of credit card fraud detection. The primary objective is to implement a sophisticated, model-based sampling technique using a **Gaussian Mixture Model (GMM)** to generate synthetic data for the minority (fraudulent) class.

The performance of this GMM-based approach is then rigorously evaluated and compared against a baseline model and other traditional resampling techniques (like SMOTE, CBU, and CBO) from a previous analysis (A3).

---

## 2. The Mission (Problem Statement)

As a data scientist for a financial institution, the mission is to build an effective fraud detection model. The core challenge lies in the provided dataset, which is highly imbalanced—fraudulent transactions represent a tiny fraction (approx. 0.17%) of all records.

The goal is to develop a data generation pipeline that allows a classifier to learn the complex patterns of the minority class, enabling it to catch a high number of fraudulent transactions (**high recall**) without excessively flagging legitimate ones (**high precision**).

---

## 3. The Blueprint (Dataset)

The project utilizes the **Credit Card Fraud Detection** dataset, publicly available on Kaggle.

-   **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
-   **Features:** The dataset contains 30 numerical features, which are the result of a PCA transformation on the original transaction data (`V1` through `V28`), along with `Time` and `Amount`.
-   **Target:** The `Class` column, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.
-   **Challenge:** Extreme class imbalance, with only 492 frauds out of 284,807 transactions.

---

## 4. Arsenal (Dependencies)

To run the analysis, the following Python libraries are required:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `imblearn` (imbalanced-learn)
-   `matplotlib`
-   `seaborn`

You can install all dependencies using the following command:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

---

## 5. Running the Operation (Usage)

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Download the dataset:**
    -   Download `creditcard.csv` from the Kaggle link provided above.
    -   Place the `creditcard.csv` file in the root directory of the project.
3.  **Launch Jupyter Notebook:**
    -   Run the `DA5401_A4_GMM_Sampling.ipynb` notebook to execute the full analysis pipeline.

---

## 6. The Heist Plans (Methodology)

The project follows a structured, comparative approach:

1.  **Part A: Reconnaissance & Baseline (From A3)**
    -   The data is loaded and analyzed to confirm the severe class imbalance.
    -   The dataset is split into training and testing sets using a stratified split to ensure the test set accurately reflects the original data distribution.
    -   A **Logistic Regression** model is trained on the raw, imbalanced training data to establish a performance baseline.

2.  **Part B: The Ghost Protocol (GMM Implementation)**
    -   A Gaussian Mixture Model is trained *exclusively on the minority class (fraud) data* from the training set.
    -   The optimal number of Gaussian components (`k`) is determined by analyzing the **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)**.
    -   The trained GMM is used as a generative model to sample a sufficient number of new, synthetic fraud instances to create a balanced training set (**GMM Oversampling**).
    -   A hybrid approach combining **Clustering-Based Undersampling (CBU)** on the majority class and GMM sampling is also explored.

3.  **Part C: The Showdown (Evaluation & Comparison)**
    -   A new Logistic Regression classifier is trained on the GMM-balanced dataset.
    -   All models are evaluated on the **original, untouched, imbalanced test set**.
    -   Performance is measured using **Precision, Recall, and F1-Score** for the minority class, as accuracy is a misleading metric in this context.

---

