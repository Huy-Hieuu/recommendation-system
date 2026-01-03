# Recommendation System

A comprehensive recommendation system implementation with Collaborative Filtering (CF) and Content-Based Filtering (CBF) algorithms using TensorFlow.

## Project Structure

```
recommendation-system/
├── CF/                          # Collaborative Filtering
│   ├── __init__.py
│   ├── collaborative_filtering.py    # NCF and MF models
│   ├── data_utils.py                  # Data preprocessing for CF
│   ├── train.py                       # Training script
│   └── predict.py                     # Prediction utilities
│
├── CBF/                         # Content-Based Filtering
│   ├── __init__.py
│   ├── content_based.py               # Content-based models
│   ├── content_data_utils.py          # Feature extraction utilities
│   ├── train_content_based.py         # Training script
│   └── predict_content_based.py       # Prediction utilities
│
├── ml-100k/                     # MovieLens 100k dataset
│   ├── u.data                   # User-item interactions
│   ├── u.item                   # Item features (genres, metadata)
│   └── ...
│
├── train_cf.py                  # Runner script for CF training
├── train_cbf.py                 # Runner script for CBF training
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Collaborative Filtering (CF)

Train a collaborative filtering model:

```bash
python train_cf.py
```

Or use as a module:

```python
from CF import create_ncf_model, compile_model
from CF.train import prepare_data_pipeline

# Prepare data
data_info = prepare_data_pipeline('ml-100k/u.data')

# Create and train model
model = create_ncf_model(
    num_users=data_info['num_users'],
    num_items=data_info['num_items']
)
model = compile_model(model)
# ... training code
```

### Content-Based Filtering (CBF)

Train a content-based model:

```bash
python train_cbf.py
```

Or use as a module:

```python
from CBF import create_content_based_model, compile_content_model
from CBF.train_content_based import prepare_content_based_pipeline

# Prepare data
data_info = prepare_content_based_pipeline(
    'ml-100k/u.data',
    'ml-100k/u.item'
)

# Create and train model
model = create_content_based_model(
    num_users=data_info['num_users'],
    item_feature_dim=data_info['item_feature_dim']
)
model = compile_content_model(model)
# ... training code
```

## Algorithms

### Collaborative Filtering (CF)
- **Neural Collaborative Filtering (NCF)**: Deep learning model with embedding layers and neural networks
- **Matrix Factorization (MF)**: Lightweight model using dot product of embeddings

### Content-Based Filtering (CBF)
- **Content-Based Model**: Deep learning model using item features (genres, metadata)
- **Simple Content Model**: Lightweight model using cosine similarity

## Features

- Modular design with clear separation of concerns
- Both CF and CBF implementations
- Support for MovieLens dataset
- Comprehensive data preprocessing utilities
- Prediction and recommendation functions
- Model training with callbacks and early stopping

## Requirements

- Python 3.8+
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- scikit-learn >= 1.3.0

