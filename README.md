# Predicting Polymer Morphologies

A machine learning framework for predicting polymer morphologies formed through Shear-Driven Polymer Precipitation (SDPP) using experimental data and computational analysis.

## Overview

This project implements machine learning models to predict polymer morphologies based on experimental parameters from shear-driven polymer precipitation processes. The work builds upon research on soft dendritic microparticles with fractal branching and nanofibrillar contact splitting that exhibit gelation at very low volume fractions, strong interparticle adhesion and binding into coatings and non-woven sheets.

## Background

Polymer morphology plays a crucial role in determining material properties and performance. The interplay between morphology, excluded volume and adhesivity of particles critically determines the physical properties of numerous soft materials and coatings. Understanding and predicting these morphological features is essential for:

- Material design and optimization
- Property prediction for new polymer compositions
- Process parameter optimization
- Quality control in manufacturing

## Features

- **Data Integration**: Combines three datasets (SDPP experimental data, solvent properties, polymer properties)
- **Feature Engineering**: Calculates Hansen Solubility Parameters (HSP), volume fractions, viscosity ratios, and relative energy difference (RED)
- **Machine Learning Models**: Implementation of 24+ ML algorithms including classifiers and regressors
- **Morphology Prediction**: Predicts polymer morphologies (1-5 categories) from processing parameters
- **Hansen Solubility Analysis**: Uses HSP theory to understand polymer-solvent interactions

## Installation

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- xgboost
- lightgbm
- catboost

### Setup

```bash
git clone https://github.com/JerushC/Predicting-Polymer-Morphologies.git
cd Predicting-Polymer-Morphologies
pip install pandas numpy scikit-learn seaborn matplotlib xgboost lightgbm catboost
```

## Usage

### Basic Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and combine datasets
sdpp = pd.read_csv('Shear-Driven_Polymer_Precipitation.csv')
solvent_df = pd.read_csv('Solvent_Information.csv')
polymer_df = pd.read_csv('Polymer_Information.csv')

# Merge datasets
combo = pd.merge(left=sdpp, right=solvent_df, how='inner', on='Solvent')
combined = pd.merge(left=combo, right=polymer_df, how='inner', on='Polymer')

# Feature engineering (HSP calculations, volume fractions, etc.)
# ... (see full implementation in sdpp_testing.py)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
rf = RandomForestClassifier(max_depth=5, n_estimators=10)
rf.fit(X_train, y_train)
```

### Model Comparison

The project evaluates 24 different machine learning algorithms:

**Classifiers:**
- K-Nearest Neighbors
- Support Vector Machines (Linear & RBF)
- Decision Tree
- Random Forest
- Neural Networks (MLP)
- AdaBoost
- Naive Bayes
- Logistic Regression
- Gradient Boosting
- LightGBM
- XGBoost

**Regressors:**
- Support Vector Regression
- Kernel Ridge
- Elastic Net
- Bayesian Ridge
- Gradient Boosting Regressor
- CatBoost Regressor

## Dataset

The project uses three integrated datasets:

### 1. Shear-Driven Polymer Precipitation (`Shear-Driven_Polymer_Precipitation.csv`)
- Polymer types and molecular weights
- Concentration levels
- Solvent and nonsolvent combinations
- **Target variable**: Morphology (categorical: 1-5)

### 2. Solvent Information (`Solvent_Information.csv`)
- Hansen Solubility Parameters (HSPd, HSPp, HSPh)
- Physical properties (viscosity, density, molecular weight)

### 3. Polymer Information (`Polymer_Information.csv`)
- Polymer Hansen Solubility Parameters
- Density and interaction radius (R0) values

### Key Features Generated:
- **Volume fractions**: Polymer and solvent volume fractions
- **HSP mixing rules**: Polymer-solution Hansen parameters
- **Viscosity relationships**: Injection viscosity and ratios
- **Energy parameters**: Relative Energy Difference (RED)

## Models

### Implemented Algorithms

The framework tests 24 different machine learning approaches, comparing their effectiveness for morphology prediction. Models are evaluated across 5 training cycles to ensure robust performance assessment.

Key algorithms include:
- **Tree-based methods**: Random Forest, Decision Trees, Gradient Boosting
- **Neural networks**: Multi-layer Perceptron with various architectures  
- **Support Vector approaches**: SVM with linear and RBF kernels, SVR
- **Ensemble methods**: AdaBoost, LightGBM, XGBoost, CatBoost
- **Probabilistic models**: Naive Bayes, Gaussian Process
- **Linear models**: Logistic Regression, Elastic Net, Bayesian Ridge

## Results

The models demonstrate varying predictive performance across different polymer morphology categories. Notable findings include:

- **Baseline accuracy**: Random classification would achieve ~20% accuracy (1 in 5 categories)
- **Model performance**: Several algorithms achieve >50% accuracy with limited training data
- **Feature importance**: Hansen Solubility Parameters and viscosity ratios show strong predictive power
- **Data preprocessing**: StandardScaler normalization improves model performance significantly

## Applications

This framework can be applied to:

- **Process Optimization**: Predicting optimal polymer precipitation conditions
- **Materials Design**: Understanding solvent-polymer compatibility for desired morphologies  
- **Quality Control**: Rapid morphology assessment in manufacturing
- **Research Support**: Hypothesis generation for SDPP experiments

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{predicting_polymer_morphologies,
  title={Predicting Polymer Morphologies via Shear-Driven Polymer Precipitation},
  author={JerushC},
  year={2024},
  url={https://github.com/JerushC/Predicting-Polymer-Morphologies}
}
```

### Related Work

This project builds upon research described in:

**Roh, S., Williams, A.H., Bang, R.S. et al.** Soft dendritic microparticles with unusual adhesion and structuring properties. *Nat. Mater.* **18**, 1315–1320 (2019). https://doi.org/10.1038/s41563-019-0508-z

## License

This project is for academic and research purposes only. All rights reserved. Contact the author for permission regarding commercial use or redistribution.

## Acknowledgments

- Rachel Bang, graduate mentor for this research project
- Nature Materials research on dendritic microparticles for morphological insights
- University research program supporting this work
- Hansen Solubility Parameter theory and applications

## Contact

For questions, issues, or collaborations, please:
- Open an issue on GitHub
- Contact the maintainer directly
