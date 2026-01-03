"""
Training script for content-based recommendation models.

This script demonstrates how to train content-based models using
item features (e.g., genres, metadata) from the MovieLens dataset.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional

try:
    from .content_based import (
        create_content_based_model,
        create_simple_content_model,
        compile_content_model
    )
    from .data_utils import (
        load_movielens_items,
        extract_genre_features,
        normalize_features,
        create_item_feature_map,
        prepare_content_based_data,
        generate_negative_samples_content,
        split_content_data,
        create_content_tf_dataset
    )
    from ..CF.data_utils import (
        encode_user_item_ids,
        create_rating_column
    )
except ImportError:
    from content_based import (
        create_content_based_model,
        create_simple_content_model,
        compile_content_model
    )
    from data_utils import (
        load_movielens_items,
        extract_genre_features,
        normalize_features,
        create_item_feature_map,
        prepare_content_based_data,
        generate_negative_samples_content,
        split_content_data,
        create_content_tf_dataset
    )
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CF'))
    from data_utils import (
        encode_user_item_ids,
        create_rating_column
    )


def prepare_content_based_pipeline(
    interactions_path: str,
    items_path: str,
    num_negative_samples: int = 1,
    test_size: float = 0.2,
    normalize: str = 'l2'
) -> Dict:
    """
    Complete data preparation pipeline for content-based models.
    
    Args:
        interactions_path: Path to user-item interactions CSV
        items_path: Path to MovieLens u.item file
        num_negative_samples: Number of negative samples per positive
        test_size: Proportion of data for testing
        normalize: Feature normalization method ('l2', 'l1', 'max', or None)
    
    Returns:
        Dictionary containing:
        - train_dataset: Training TensorFlow dataset
        - test_dataset: Testing TensorFlow dataset
        - num_users: Number of unique users
        - item_feature_dim: Dimension of item features
        - item_feature_map: Mapping from item_id to feature vector
        - mappings: ID encoding mappings
    """
    # Load interactions
    print("Loading interactions...")
    # MovieLens u.data is tab-separated: user_id, item_id, rating, timestamp
    interactions_df = pd.read_csv(
        interactions_path,
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    # Encode user and item IDs
    print("Encoding user and item IDs...")
    interactions_encoded, mappings, counts = encode_user_item_ids(interactions_df)
    
    # Create rating column (binary)
    interactions_encoded = create_rating_column(interactions_encoded)
    
    # Load item features
    print("Loading item features...")
    items_df = load_movielens_items(items_path)
    
    # Extract genre features
    print("Extracting genre features...")
    feature_matrix, feature_names = extract_genre_features(items_df)
    print(f"Feature dimension: {feature_matrix.shape[1]}")
    print(f"Features: {feature_names}")
    
    # Normalize features
    if normalize:
        print(f"Normalizing features using {normalize}...")
        feature_matrix = normalize_features(feature_matrix, method=normalize)
    
    # Create item feature mapping
    item_feature_map = create_item_feature_map(items_df, feature_matrix)
    
    # Prepare positive samples
    print("Preparing positive samples...")
    user_ids, item_features, ratings = prepare_content_based_data(
        interactions_encoded,
        item_feature_map,
        user_col='user_encoded',
        item_col='item_encoded',
        rating_col='rating'
    )
    
    # Generate negative samples
    if num_negative_samples > 0:
        print(f"Generating {num_negative_samples} negative samples per positive...")
        neg_user_ids, neg_item_features, neg_ratings = \
            generate_negative_samples_content(
                user_ids,
                item_features,
                feature_matrix,
                num_negative=num_negative_samples
            )
        
        # Combine positive and negative
        user_ids = np.concatenate([user_ids, neg_user_ids])
        item_features = np.concatenate([item_features, neg_item_features])
        ratings = np.concatenate([ratings, neg_ratings])
    
    # Convert ratings to binary if needed
    if ratings.max() > 1.0:
        threshold = np.median(ratings[ratings > 0])
        ratings = (ratings >= threshold).astype(float)
    
    # Split into train/test
    print("Splitting data into train/test...")
    train_user, train_features, train_rating, test_user, test_features, test_rating = \
        split_content_data(user_ids, item_features, ratings, test_size=test_size)
    
    print(f"Training samples: {len(train_user)}")
    print(f"Test samples: {len(test_user)}")
    
    # Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_content_tf_dataset(
        train_user, train_features, train_rating,
        batch_size=32,
        shuffle=True
    )
    
    test_dataset = create_content_tf_dataset(
        test_user, test_features, test_rating,
        batch_size=32,
        shuffle=False
    )
    
    # Prepare dataset for model input format
    def map_to_inputs(batch):
        return (
            (batch['user_id'], batch['item_features']),
            batch['rating']
        )
    
    train_dataset = train_dataset.map(map_to_inputs)
    test_dataset = test_dataset.map(map_to_inputs)
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_users': counts['num_users'],
        'item_feature_dim': feature_matrix.shape[1],
        'item_feature_map': item_feature_map,
        'mappings': mappings,
        'feature_names': feature_names
    }


def train_content_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    epochs: int = 10,
    validation_freq: int = 1,
    callbacks: Optional[list] = None
) -> keras.callbacks.History:
    """
    Train a content-based recommendation model.
    
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
    Main training function for content-based model.
    """
    # MovieLens dataset paths
    # Try relative path first (when run from root), then absolute
    import os
    if os.path.exists('ml-100k/u.data'):
        interactions_path = 'ml-100k/u.data'
        items_path = 'ml-100k/u.item'
    else:
        interactions_path = '../ml-100k/u.data'
        items_path = '../ml-100k/u.item'
    
    # Prepare data
    print("=" * 60)
    print("Content-Based Recommendation System Training")
    print("=" * 60)
    
    data_info = prepare_content_based_pipeline(
        interactions_path,
        items_path,
        num_negative_samples=1,
        test_size=0.2,
        normalize='l2'
    )
    
    num_users = data_info['num_users']
    item_feature_dim = data_info['item_feature_dim']
    
    print(f"\nModel Configuration:")
    print(f"  Number of users: {num_users}")
    print(f"  Item feature dimension: {item_feature_dim}")
    print(f"  Features: {', '.join(data_info['feature_names'])}")
    
    # Create and compile content-based model
    print("\n" + "=" * 60)
    print("Creating Content-Based Model...")
    print("=" * 60)
    
    content_model = create_content_based_model(
        num_users=num_users,
        item_feature_dim=item_feature_dim,
        user_embedding_dim=50,
        feature_hidden_layers=[128, 64],
        combined_hidden_layers=[64, 32, 16],
        dropout_rate=0.2
    )
    
    content_model = compile_content_model(content_model, learning_rate=0.001)
    content_model.build(input_shape=[(None,), (None, item_feature_dim)])
    content_model.summary()
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Content-Based Model...")
    print("=" * 60)
    
    history = train_content_model(
        content_model,
        data_info['train_dataset'],
        data_info['test_dataset'],
        epochs=15
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)
    
    test_loss, test_accuracy, test_precision, test_recall = content_model.evaluate(
        data_info['test_dataset'],
        verbose=0
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    
    # Save model
    model_path = 'content_based_model.keras'
    content_model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Example: Create and train simple content model
    print("\n" + "=" * 60)
    print("Creating Simple Content-Based Model...")
    print("=" * 60)
    
    simple_model = create_simple_content_model(
        num_users=num_users,
        item_feature_dim=item_feature_dim,
        embedding_dim=50
    )
    
    simple_model = compile_content_model(simple_model, learning_rate=0.001)
    simple_model.build(input_shape=[(None,), (None, item_feature_dim)])
    
    print("\nTraining Simple Content-Based Model...")
    simple_history = train_content_model(
        simple_model,
        data_info['train_dataset'],
        data_info['test_dataset'],
        epochs=15
    )
    
    simple_test_loss, simple_test_accuracy, _, _ = simple_model.evaluate(
        data_info['test_dataset'],
        verbose=0
    )
    
    print(f"\nSimple Model Test Results:")
    print(f"  Loss: {simple_test_loss:.4f}")
    print(f"  Accuracy: {simple_test_accuracy:.4f}")
    
    simple_model_path = 'simple_content_model.keras'
    simple_model.save(simple_model_path)
    print(f"\nSimple model saved to {simple_model_path}")


if __name__ == '__main__':
    main()

