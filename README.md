# Microscopic Diagnosis of Leukemia using Deep Learning & Statistical Optimization

## Project Overview
Acute Lymphoblastic Leukemia (ALL) diagnosis is time-sensitive and prone to observer bias. This project develops a computer-vision framework to classify leukemic blasts vs. normal cells from microscopic images. 

**Key Challenge:** The C-NMC dataset suffered from severe **Class Imbalance** ($N_{normal} \gg N_{cancer}$), posing a risk of statistical bias where the model prioritizes the majority class.

**Objective:** To design a robust classifier that maximizes **Sensitivity (Recall)**, ensuring minimal false-negative diagnoses which are critical in clinical screening.

## Biostatistical Methodology
This project integrates deep learning with rigorous statistical sampling techniques to ensure validity:

### 1. Cohort Construction via Stratified Sampling
Instead of simple random splitting, **Stratified Random Sampling** was implemented to partition the dataset.
- **Goal:** To ensure consistent disease prevalence (Prior Probability) across Training, Validation, and Test sets.
- **Impact:** Prevents **Distribution Shift** and ensures that the evaluation metrics reflect true diagnostic performance unbiased by sample variance.

### 2. Bias Mitigation via Under-sampling
To address the selection bias inherent in the imbalanced dataset:
- **Technique:** Algorithmic **Under-sampling** was applied to the majority class (Normal cells).
- **Goal:** To achieve a balanced 50/50 case-control distribution during training.
- **Impact:** Forces the model to minimize Bayesian Risk by learning conditional features $P(X|Y)$ rather than relying on the skewed prior $P(Y)$.

### 3. Biological Variability Simulation
**Data Augmentation** (rotation, flipping, shifting) was employed not just to increase data size, but to simulate real-world **biological variability** and microscopy artifacts, enhancing the model's robustness and generalization.

## Model Architecture
- **Backbone:** EfficientNetB3 (Transfer Learning from ImageNet).
- **Optimization:** A custom **Learning Rate Adjuster (LRA)** callback was developed to dynamically tune the learning rate based on the stability of the validation loss, ensuring convergence to a global minimum without overfitting.

## Results
The model achieved state-of-the-art diagnostic metrics on the test cohort:
- **F1-Score:** 0.97
- **Recall (Sensitivity):** 96%
- **Accuracy:** 96.2%

By prioritizing Sensitivity, the model minimizes Type II Errors (False Negatives), proving its viability as a safe automated screening tool for resource-constrained clinical settings.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
