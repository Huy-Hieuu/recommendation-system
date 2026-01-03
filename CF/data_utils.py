"""
Data preprocessing utilities for collaborative filtering.

This module provides functions for loading, preprocessing, and preparing
user-item interaction data for training recommendation models.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import kagglehub

def download_data():
    path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")
    return path

def load_interaction_data(file_path: str) -> pd.DataFrame:
    """
    Load user-item interaction data from a CSV file.
    
    Expected columns: user_id, item_id, rating (or interaction)
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame with user-item interactions
    """
    df = pd.read_csv(file_path)
    required_columns = ['user_id', 'item_id']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"DataFrame must contain columns: {required_columns}. "
            f"Found: {df.columns.tolist()}"
        )
    
    return df


def encode_user_item_ids(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id'
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, int]]:
    """
    Encode user and item IDs to sequential integers starting from 0.
    
    Args:
        df: DataFrame with user and item columns
        user_col: Name of the user ID column
        item_col: Name of the item ID column
    
    Returns:
        Tuple of:
        - DataFrame with encoded IDs
        - Mapping dictionaries (original_id -> encoded_id)
        - Count dictionaries (num_users, num_items)
    """
    df = df.copy()
    
    # Create mappings
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(unique_users))}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(unique_items))}
    
    # Apply mappings
    df['user_encoded'] = df[user_col].map(user_to_idx)
    df['item_encoded'] = df[item_col].map(item_to_idx)
    
    mappings = {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx
    }
    
    counts = {
        'num_users': len(unique_users),
        'num_items': len(unique_items)
    }
    
    return df, mappings, counts


def create_rating_column(
    df: pd.DataFrame,
    rating_col: Optional[str] = None,
    default_rating: float = 1.0,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Create or normalize rating column for binary classification.
    
    If rating_col exists, converts ratings to binary (0/1) based on threshold.
    If rating_col doesn't exist, creates binary ratings (all 1.0).
    
    Args:
        df: DataFrame with interactions
        rating_col: Name of rating column (if exists)
        default_rating: Default rating value if no rating column
        threshold: Threshold for binary conversion (default: median)
    
    Returns:
        DataFrame with 'rating' column (binary 0/1)
    """
    df = df.copy()
    
    if rating_col and rating_col in df.columns:
        if threshold is None:
            threshold = df[rating_col].median()
        df['rating'] = (df[rating_col] >= threshold).astype(float)
    else:
        df['rating'] = default_rating
    
    return df


def generate_negative_samples(
    df: pd.DataFrame,
    num_negative: int,
    num_users: int,
    num_items: int,
    user_col: str = 'user_encoded',
    item_col: str = 'item_encoded'
) -> pd.DataFrame:
    """
    Generate negative samples (non-interacted user-item pairs).
    
    Args:
        df: DataFrame with positive interactions
        num_negative: Number of negative samples per positive sample
        num_users: Total number of users
        num_items: Total number of items
        user_col: Name of encoded user column
        item_col: Name of encoded item column
    
    Returns:
        DataFrame with negative samples (rating = 0.0)
    """
    # Create set of existing interactions for fast lookup
    existing_interactions = set(
        zip(df[user_col].values, df[item_col].values)
    )
    
    negative_samples = []
    total_negative = len(df) * num_negative
    
    np.random.seed(42)
    attempts = 0
    max_attempts = total_negative * 10
    
    while len(negative_samples) < total_negative and attempts < max_attempts:
        user_id = np.random.randint(0, num_users)
        item_id = np.random.randint(0, num_items)
        
        if (user_id, item_id) not in existing_interactions:
            negative_samples.append({
                user_col: user_id,
                item_col: item_id,
                'rating': 0.0
            })
        
        attempts += 1
    
    negative_df = pd.DataFrame(negative_samples)
    return negative_df


def prepare_training_data(
    df: pd.DataFrame,
    num_negative_samples: int = 1,
    user_col: str = 'user_encoded',
    item_col: str = 'item_encoded',
    rating_col: str = 'rating'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data with positive and negative samples.
    
    Args:
        df: DataFrame with encoded user/item IDs and ratings
        num_negative_samples: Number of negative samples per positive
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column
    
    Returns:
        Tuple of (user_ids, item_ids, ratings) as numpy arrays
    """
    # Get positive samples
    positive_df = df[df[rating_col] > 0].copy()
    
    # Generate negative samples if needed
    if num_negative_samples > 0:
        num_users = df[user_col].max() + 1
        num_items = df[item_col].max() + 1
        
        negative_df = generate_negative_samples(
            positive_df,
            num_negative_samples,
            num_users,
            num_items,
            user_col,
            item_col
        )
        
        # Combine positive and negative samples
        combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    else:
        combined_df = positive_df
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Extract arrays
    user_ids = combined_df[user_col].values
    item_ids = combined_df[item_col].values
    ratings = combined_df[rating_col].values
    
    return user_ids, item_ids, ratings


def split_train_test(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        ratings: Array of ratings
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_user, train_item, train_rating, 
                 test_user, test_item, test_rating)
    """
    train_user, test_user, train_item, test_item, train_rating, test_rating = \
        train_test_split(
            user_ids, item_ids, ratings,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
    
    return train_user, train_item, train_rating, test_user, test_item, test_rating


def create_tf_dataset(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset from user-item interaction arrays.
    
    Args:
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        ratings: Array of ratings/labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
    
    Returns:
        TensorFlow Dataset ready for training
    """
    dataset = tf.data.Dataset.from_tensor_slices({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

