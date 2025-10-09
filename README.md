# Decision Trees and Random Forests
**Course:** INF264 – Machine Learning, University of Bergen (Fall 2025)  
**Authors:** Maria Graham and Jake Menegus  

## Overview
This project implements Decision Tree and Random Forest classifiers from scratch in Python, applied to a letter recognition dataset containing 2000 samples with 16 numeric features. The objective was to explore the underlying algorithms, perform model selection, and compare the custom implementations with scikit-learn’s equivalents.

## Implementation
- Developed a Decision Tree model including custom impurity measures (Gini, entropy), feature splitting, and recursive tree construction.  
- Extended the implementation to a Random Forest using bagging, random feature subset selection, and majority voting for final predictions.  
- Employed stratified 5-fold cross-validation and systematic hyperparameter tuning for fair evaluation.

## Results
- **Best Decision Tree:** 94.33% accuracy (criterion: entropy, max depth: 10)  
- **Best Random Forest:** 97.66% accuracy (20 estimators, max depth: 10, criterion: gini, max features: sqrt)  
- Custom implementations achieved performance comparable to scikit-learn’s models.

## Feature Importance
Permutation Feature Importance was applied to evaluate the relative contribution of each feature to model performance.

## Key Findings
Random Forests achieved higher accuracy and generalization compared to single Decision Trees, primarily due to ensemble averaging and variance reduction. Fixed random seeds ensured reproducibility and unbiased evaluation.

## Future Work
Potential improvements include implementing pruning for better generalization, evaluating on additional datasets, and further analyzing model interpretability and ethical considerations.
