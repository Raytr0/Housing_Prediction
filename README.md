# Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## Overview
This project implements and evaluates advanced machine learning models to predict housing prices. The core focus is on comparing the performance of **Random Forest** and **XGBoost** (Extreme Gradient Boosting) algorithms.

By analyzing housing market data, this project aims to determine which ensemble learning technique offers better accuracy and generalization for regression tasks. The repository also includes a comprehensive **IEEE-style project report** detailing the methodology, data processing, and final results.

## Repository Structure

Housing_Prediction/
├── RandomForest.py              # Implementation of the Random Forest model  
├── XGBoost.py                   # Implementation of the XGBoost model  
├── XGBoostwithRandomForest.py   # Hybrid/Comparative script running both models  
├── ai_project_report.pdf        # Full project report (Methodology & Results)  
├── ieee-conference-template.pdf # Template used for the report  
└── README.md                    # Project documentation  

## Prerequisites & Installation

To run this project locally, you will need **Python 3.x** installed.

1. **Clone the repository:**
    
    git clone https://github.com/Raytr0/Housing_Prediction.git
    cd Housing_Prediction

2. **Install the required dependencies:**
   It is recommended to use a virtual environment. You can install the necessary libraries via pip:

    pip install numpy pandas scikit-learn xgboost matplotlib seaborn

## Usage

You can execute the models directly from the terminal. Ensure your dataset is located in the root directory (or update the file paths in the scripts).

### Run Random Forest
Trains the Random Forest regressor and outputs performance metrics.

    python RandomForest.py

### Run XGBoost
Trains the XGBoost regressor and outputs performance metrics.

    python XGBoost.py

### Run Comparative Analysis
Runs both models to compare their performance side-by-side.

    python XGBoostwithRandomForest.py

## Methodology

### 1. Random Forest
* **Type:** Bagging Ensemble.
* **Mechanism:** Constructs a multitude of decision trees at training time and outputs the mean prediction of the individual trees.
* **Advantage:** Reduces overfitting and works well with high-dimensional data.

### 2. XGBoost
* **Type:** Boosting Ensemble.
* **Mechanism:** Builds models sequentially, where each new model attempts to correct the errors of the previous ones using gradient descent optimization.
* **Advantage:** Highly efficient, handles missing data well, and often provides state-of-the-art results in regression challenges.

## Results & Analysis

For a detailed breakdown of the model performance, including **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R² Scores**, please refer to the project report:

**[View the Project Report (PDF)](./ai_project_report.pdf)**

## Authors
* **Raytr0**

## License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

