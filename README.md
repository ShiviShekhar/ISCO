# Intelligent Sales Coverage Optimization (ISCO)

This project implements an Intelligent Sales Coverage Optimization (ISCO) system to strategically assign sales specialists to accounts, maximizing sales effectiveness and revenue.

## Project Stages

The solution is divided into two core stages:

1.  **Predictive Modeling**: Train an AI model to predict a "success score" (e.g., expected booking value) for every possible seller-account pairing. This involves extensive feature engineering, particularly using NLP to create a `technology_match_score`.

2.  **Optimization**: Use the predicted success scores to generate an optimal assignment plan using Integer Linear Programming (ILP). The model maximizes overall success while respecting business constraints like seller capacity, workload balance, and geographic alignment.

## Technology Stack

-   **Python**: Core programming language
-   **Pandas**: Data manipulation and analysis
-   **Scikit-learn**: Machine learning utilities
-   **XGBoost**: Gradient boosting model for success score prediction
-   **SpaCy**: Natural Language Processing for feature engineering
-   **Google OR-Tools**: Integer Linear Programming for optimization
-   **Flask**: To build a simple API for serving the model (optional)
-   **Jupyter**: For notebooks and experimentation
