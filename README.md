# LogiChain: Supply Chain Delay Prediction

## Overview

This project develops and evaluates a machine learning model to predict late deliveries within a supply chain. By leveraging historical order and shipping data, the model aims to identify factors contributing to delays and provide proactive insights.

## Dataset

The core of this project uses a refined supply chain dataset containing various features related to orders, customers, products, and shipping logistics. Key features include:

- `shipping_mode`
- `market`
- `customer_segment`
- `department_name`
- `category_name`
- `order_region`
- `order_country`
- `sales`
- `benefit_per_order`
- `order_item_quantity`
- `product_price`
- `days_for_shipment_scheduled`
- `latitude_src`, `longitude_src` (Source location coordinates)
- `latitude_dest`, `longitude_dest` (Destination location coordinates)
- `distance` (Calculated distance between source and destination)

The target variable for prediction is `late_delivery_risk`, a binary indicator (0 for on-time, 1 for late).

## Methodology

1.  **Data Ingestion and Preparation:**
    *   ZIP files containing CSV datasets are uploaded and extracted.
    *   Individual CSVs are loaded into Pandas DataFrames.
    *   Datasets with identical columns are combined where appropriate.

2.  **Feature Engineering:**
    *   A `distance` feature is engineered by calculating the Euclidean distance between `latitude_src`, `longitude_src` and `latitude_dest`, `longitude_dest`.

3.  **Categorical Feature Encoding:**
    *   Label Encoding is applied to convert categorical features (`shipping_mode`, `market`, `customer_segment`, `department_name`, `category_name`, `order_region`, `order_country`) into numerical representations.

4.  **Model Training:**
    *   The dataset is split into training and testing sets using `train_test_split` with stratification to maintain the proportion of late deliveries.
    *   An `XGBoost Classifier` is used for training, configured with `n_estimators=500`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, and `random_state=42`.
    *   The model is trained on the prepared features to predict `late_delivery_risk`.

5.  **Evaluation:**
    *   The model's performance is evaluated using `accuracy_score`, `classification_report`, and `confusion_matrix` on the test set.

6.  **Model Persistence:**
    *   The trained XGBoost model and the LabelEncoders (for categorical features) are saved using `joblib` to allow for future deployment and inference without retraining.

## Installation

To run this project, you'll need the following libraries:

```bash
pip install pandas scikit-learn xgboost numpy joblib
