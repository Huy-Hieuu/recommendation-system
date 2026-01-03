"""
Data preprocessing utilities for content-based recommendation systems.

This module provides functions for extracting and processing item features
for content-based recommendation models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_movielens_items(file_path: str) -> pd.DataFrame:
    """
    Load MovieLens item data with genre features.
    
    Expected format: item_id|title|release_date|video_release_date|url|genre_binary...
    
    Args:
        file_path: Path to u.item file
    
    Returns:
        DataFrame with item_id and feature columns
    """
    columns = ['item_id', 'title', 'release_date', 'video_release_date', 'url']
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    columns.extend(genre_columns)
    
    df = pd.read_csv(
        file_path,
        sep='|',
        header=None,
        names=columns,
        encoding='latin-1'
    )
    
    return df


def extract_genre_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract genre features from MovieLens item DataFrame.
    
    Args:
        df: DataFrame with genre columns
    
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    available_genres = [col for col in genre_columns if col in df.columns]
    feature_matrix = df[available_genres].values.astype(float)
    
    return feature_matrix, available_genres


def normalize_features(features: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize feature vectors.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        method: Normalization method ('l2', 'l1', 'max', or None)
    
    Returns:
        Normalized feature matrix
    """
    if method is None:
        return features
    
    if method == 'l2':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return features / norms
    elif method == 'l1':
        norms = np.sum(np.abs(features), axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return features / norms
    elif method == 'max':
        max_vals = np.max(np.abs(features), axis=1, keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)
        return features / max_vals
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_item_feature_map(
    item_df: pd.DataFrame,
    feature_matrix: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Create mapping from item_id to feature vector.
    
    Args:
        item_df: DataFrame with item_id column
        feature_matrix: Feature matrix aligned with item_df
    
    Returns:
        Dictionary mapping item_id -> feature_vector
    """
    item_feature_map = {}
    
    for idx, item_id in enumerate(item_df['item_id'].values):
        item_feature_map[item_id] = feature_matrix[idx]
    
    return item_feature_map


def prepare_content_based_data(
    interactions_df: pd.DataFrame,
    item_feature_map: Dict[int, np.ndarray],
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    rating_col: str = 'rating'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for content-based model training.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        item_feature_map: Dictionary mapping item_id -> feature_vector
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column
    
    Returns:
        Tuple of (user_ids, item_features, ratings) as numpy arrays
    """
    user_ids = []
    item_features = []
    ratings = []
    
    for _, row in interactions_df.iterrows():
        user_id = row[user_col]
        item_id = row[item_col]
        rating = row[rating_col]
        
        if item_id in item_feature_map:
            user_ids.append(user_id)
            item_features.append(item_feature_map[item_id])
            ratings.append(rating)
    
    return (
        np.array(user_ids),
        np.array(item_features),
        np.array(ratings)
    )


def generate_negative_samples_content(
    positive_user_ids: np.ndarray,
    positive_item_features: np.ndarray,
    all_item_features: np.ndarray,
    num_negative: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate negative samples for content-based training.
    
    Args:
        positive_user_ids: Array of user IDs from positive interactions
        positive_item_features: Array of item features from positive interactions
        all_item_features: Array of all available item features
        num_negative: Number of negative samples per positive
    
    Returns:
        Tuple of (user_ids, item_features, ratings) for negative samples
    """
    negative_user_ids = []
    negative_item_features = []
    negative_ratings = []
    
    np.random.seed(42)
    num_all_items = len(all_item_features)
    
    for user_id, _ in zip(positive_user_ids, positive_item_features):
        for _ in range(num_negative):
            random_item_idx = np.random.randint(0, num_all_items)
            negative_user_ids.append(user_id)
            negative_item_features.append(all_item_features[random_item_idx])
            negative_ratings.append(0.0)
    
    return (
        np.array(negative_user_ids),
        np.array(negative_item_features),
        np.array(negative_ratings)
    )


def split_content_data(
    user_ids: np.ndarray,
    item_features: np.ndarray,
    ratings: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split content-based data into training and testing sets.
    
    Args:
        user_ids: Array of user IDs
        item_features: Array of item feature vectors
        ratings: Array of ratings
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_user, train_features, train_rating,
                 test_user, test_features, test_rating)
    """
    train_user, test_user, train_features, test_features, train_rating, test_rating = \
        train_test_split(
            user_ids, item_features, ratings,
            test_size=test_size,
            random_state=random_state
        )
    
    return train_user, train_features, train_rating, test_user, test_features, test_rating


def create_content_tf_dataset(
    user_ids: np.ndarray,
    item_features: np.ndarray,
    ratings: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset for content-based training.
    
    Args:
        user_ids: Array of user IDs
        item_features: Array of item feature vectors
        ratings: Array of ratings/labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
    
    Returns:
        TensorFlow Dataset ready for training
    """
    dataset = tf.data.Dataset.from_tensor_slices({
        'user_id': user_ids,
        'item_features': item_features,
        'rating': ratings
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def build_user_profile(
    user_interactions: pd.DataFrame,
    item_feature_map: Dict[int, np.ndarray],
    user_id: int,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    rating_col: str = 'rating',
    aggregation: str = 'weighted_mean'
) -> np.ndarray:
    """
    Build user preference profile from interaction history.
    
    Args:
        user_interactions: DataFrame with all user interactions
        item_feature_map: Dictionary mapping item_id -> feature_vector
        user_id: ID of user to build profile for
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column
        aggregation: Aggregation method ('weighted_mean', 'mean', 'max')
    
    Returns:
        User preference vector (same dimension as item features)
    """
    user_data = user_interactions[user_interactions[user_col] == user_id]
    
    if len(user_data) == 0:
        raise ValueError(f"No interactions found for user {user_id}")
    
    feature_vectors = []
    weights = []
    
    for _, row in user_data.iterrows():
        item_id = row[item_col]
        rating = row[rating_col]
        
        if item_id in item_feature_map:
            feature_vectors.append(item_feature_map[item_id])
            weights.append(rating)
    
    if not feature_vectors:
        raise ValueError(f"No valid item features found for user {user_id}")
    
    feature_matrix = np.array(feature_vectors)
    weights_array = np.array(weights)
    
    if aggregation == 'weighted_mean':
        weights_normalized = weights_array / np.sum(weights_array)
        user_profile = np.average(feature_matrix, axis=0, weights=weights_normalized)
    elif aggregation == 'mean':
        user_profile = np.mean(feature_matrix, axis=0)
    elif aggregation == 'max':
        user_profile = np.max(feature_matrix, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return user_profile

