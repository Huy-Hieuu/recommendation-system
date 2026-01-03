"""
Training script for collaborative filtering recommendation models.

This script demonstrates how to train Neural Collaborative Filtering (NCF)
and Matrix Factorization models on user-item interaction data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional

try:
    from .collaborative_filtering import (
        create_ncf_model,
        create_mf_model,
        compile_model
    )
    from .data_utils import (
        load_interaction_data,
        encode_user_item_ids,
        create_rating_column,
        prepare_training_data,
        split_train_test,
        create_tf_dataset
    )
except ImportError:
    from collaborative_filtering import (
        create_ncf_model,
        create_mf_model,
        compile_model
    )
    from data_utils import (
        load_interaction_data,
        encode_user_item_ids,
        create_rating_column,
        prepare_training_data,
        split_train_test,
        create_tf_dataset
    )


def prepare_data_pipeline(
    data_path: str,
    num_negative_samples: int = 1,
    test_size: float = 0.2
) -> Dict:
    """
    Complete data preparation pipeline from raw data to training datasets.
    
    Args:
        data_path: Path to CSV file with user-item interactions
        num_negative_samples: Number of negative samples per positive
        test_size: Proportion of data for testing
    
    Returns:
        Dictionary containing:
        - train_dataset: Training TensorFlow dataset
        - test_dataset: Testing TensorFlow dataset
        - num_users: Number of unique users
        - num_items: Number of unique items
        - mappings: ID encoding mappings
    """
    # Load data
    df = load_interaction_data(data_path)
    
    # Encode user and item IDs
    df_encoded, mappings, counts = encode_user_item_ids(df)
    
    # Create rating column (binary)
    df_encoded = create_rating_column(df_encoded)
    
    # Prepare training data with negative samples
    user_ids, item_ids, ratings = prepare_training_data(
        df_encoded,
        num_negative_samples=num_negative_samples
    )
    
    # Split into train/test
    train_user, train_item, train_rating, test_user, test_item, test_rating = \
        split_train_test(user_ids, item_ids, ratings, test_size=test_size)
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(
        train_user, train_item, train_rating,
        batch_size=32,
        shuffle=True
    )
    
    test_dataset = create_tf_dataset(
        test_user, test_item, test_rating,
        batch_size=32,
        shuffle=False
    )
    
    # Prepare dataset for model input format
    def map_to_inputs(batch):
        return (
            (batch['user_id'], batch['item_id']),
            batch['rating']
        )
    
    train_dataset = train_dataset.map(map_to_inputs)
    test_dataset = test_dataset.map(map_to_inputs)
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_users': counts['num_users'],
        'num_items': counts['num_items'],
        'mappings': mappings
    }


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    epochs: int = 10,
    validation_freq: int = 1,
    callbacks: Optional[list] = None
) -> keras.callbacks.History:
    """
    Train a collaborative filtering model.
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        test_dataset: Validation/test dataset
        epochs: Number of training epochs
        validation_freq: Frequency of validation
        callbacks: List of Keras callbacks
    
    Returns:
        Training history
    """
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
    
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        validation_freq=validation_freq,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def main():
    """
    Main training function demonstrating usage.
    """
    # Example: You would replace this with your actual data path
    # data_path = 'path/to/your/interactions.csv'
    
    # For demonstration, we'll create synthetic data
    print("Creating synthetic data for demonstration...")
    np.random.seed(42)
    num_users = 1000
    num_items = 500
    num_interactions = 10000
    
    synthetic_data = {
        'user_id': np.random.randint(0, num_users, num_interactions),
        'item_id': np.random.randint(0, num_items, num_interactions),
        'rating': np.random.choice([1, 2, 3, 4, 5], num_interactions)
    }
    
    import pandas as pd
    df = pd.DataFrame(synthetic_data)
    data_path = 'synthetic_interactions.csv'
    df.to_csv(data_path, index=False)
    print(f"Synthetic data saved to {data_path}")
    
    # Prepare data
    print("\nPreparing data...")
    data_info = prepare_data_pipeline(
        data_path,
        num_negative_samples=1,
        test_size=0.2
    )
    
    num_users = data_info['num_users']
    num_items = data_info['num_items']
    
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    
    # Create and compile NCF model
    print("\nCreating NCF model...")
    ncf_model = create_ncf_model(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=50,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2
    )
    
    ncf_model = compile_model(ncf_model, learning_rate=0.001)
    ncf_model.build(input_shape=[(None,), (None,)])
    ncf_model.summary()
    
    # Train NCF model
    print("\nTraining NCF model...")
    history = train_model(
        ncf_model,
        data_info['train_dataset'],
        data_info['test_dataset'],
        epochs=10
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = ncf_model.evaluate(
        data_info['test_dataset'],
        verbose=0
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    
    # Save model
    model_path = 'ncf_model.keras'
    ncf_model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Example: Create and train MF model
    print("\n" + "="*50)
    print("Creating Matrix Factorization model...")
    mf_model = create_mf_model(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=50
    )
    
    mf_model = compile_model(mf_model, learning_rate=0.001)
    mf_model.build(input_shape=[(None,), (None,)])
    
    print("\nTraining MF model...")
    mf_history = train_model(
        mf_model,
        data_info['train_dataset'],
        data_info['test_dataset'],
        epochs=10
    )
    
    mf_test_loss, mf_test_accuracy, _, _ = mf_model.evaluate(
        data_info['test_dataset'],
        verbose=0
    )
    
    print(f"\nMF Model Test Results:")
    print(f"  Loss: {mf_test_loss:.4f}")
    print(f"  Accuracy: {mf_test_accuracy:.4f}")


if __name__ == '__main__':
    main()

