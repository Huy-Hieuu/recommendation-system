"""
Prediction and recommendation utilities for content-based models.

This module provides functions for making predictions and generating
recommendations using trained content-based recommendation models.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import pandas as pd

try:
    from .data_utils import (
        load_movielens_items,
        extract_genre_features,
        normalize_features,
        create_item_feature_map,
        build_user_profile
    )
except ImportError:
    from data_utils import (
        load_movielens_items,
        extract_genre_features,
        normalize_features,
        create_item_feature_map,
        build_user_profile
    )


def predict_user_item_rating_content(
    model: tf.keras.Model,
    user_id: int,
    item_features: np.ndarray
) -> float:
    """
    Predict rating for a specific user-item pair using item features.
    
    Args:
        model: Trained content-based model
        user_id: Encoded user ID
        item_features: Item feature vector
    
    Returns:
        Predicted rating score
    """
    user_tensor = tf.constant([[user_id]], dtype=tf.int32)
    item_tensor = tf.constant([item_features], dtype=tf.float32)
    
    prediction = model((user_tensor, item_tensor), training=False)
    return float(prediction.numpy()[0][0])


def get_content_based_recommendations(
    model: tf.keras.Model,
    user_id: int,
    item_feature_map: Dict[int, np.ndarray],
    top_k: int = 10,
    exclude_interacted: Optional[List[int]] = None
) -> List[Tuple[int, float]]:
    """
    Get top-k item recommendations for a user based on content features.
    
    Args:
        model: Trained content-based model
        user_id: Encoded user ID
        item_feature_map: Dictionary mapping item_id -> feature_vector
        top_k: Number of top recommendations to return
        exclude_interacted: List of item IDs to exclude (already interacted)
    
    Returns:
        List of (item_id, score) tuples, sorted by score descending
    """
    if exclude_interacted:
        candidate_items = {
            item_id: features
            for item_id, features in item_feature_map.items()
            if item_id not in exclude_interacted
        }
    else:
        candidate_items = item_feature_map
    
    if not candidate_items:
        return []
    
    # Batch predictions for efficiency
    item_ids = list(candidate_items.keys())
    item_features = np.array([candidate_items[item_id] for item_id in item_ids])
    
    user_ids = np.array([user_id] * len(item_ids))
    user_tensor = tf.constant(user_ids, dtype=tf.int32)
    item_tensor = tf.constant(item_features, dtype=tf.float32)
    
    predictions = model((user_tensor, item_tensor), training=False)
    scores = predictions.numpy().flatten()
    
    # Get top-k items
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    recommendations = [
        (item_ids[idx], float(scores[idx]))
        for idx in top_indices
    ]
    
    return recommendations


def find_similar_items_by_content(
    item_features: np.ndarray,
    item_feature_map: Dict[int, np.ndarray],
    top_k: int = 10,
    exclude_item_id: Optional[int] = None
) -> List[Tuple[int, float]]:
    """
    Find items similar to a given item based on content features.
    
    Uses cosine similarity between feature vectors.
    
    Args:
        item_features: Feature vector of the reference item
        item_feature_map: Dictionary mapping item_id -> feature_vector
        top_k: Number of similar items to return
        exclude_item_id: Item ID to exclude from results
    
    Returns:
        List of (item_id, similarity_score) tuples
    """
    # Normalize reference item features
    item_norm = item_features / (np.linalg.norm(item_features) + 1e-8)
    
    similarities = []
    
    for item_id, features in item_feature_map.items():
        if exclude_item_id is not None and item_id == exclude_item_id:
            continue
        
        # Compute cosine similarity
        features_norm = features / (np.linalg.norm(features) + 1e-8)
        similarity = np.dot(item_norm, features_norm)
        similarities.append((item_id, float(similarity)))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def get_user_preference_vector(
    model: tf.keras.Model,
    user_id: int
) -> np.ndarray:
    """
    Extract user preference vector from trained model.
    
    Args:
        model: Trained content-based model
        user_id: Encoded user ID
    
    Returns:
        User preference embedding vector
    """
    user_embedding_layer = model.get_layer('user_embedding')
    user_emb = user_embedding_layer(tf.constant([[user_id]], dtype=tf.int32))
    user_emb = tf.reshape(user_emb, [-1])
    
    return user_emb.numpy()


def recommend_by_user_profile(
    user_profile: np.ndarray,
    item_feature_map: Dict[int, np.ndarray],
    top_k: int = 10,
    exclude_interacted: Optional[List[int]] = None
) -> List[Tuple[int, float]]:
    """
    Get recommendations by computing similarity between user profile and items.
    
    Args:
        user_profile: User preference vector (from build_user_profile)
        item_feature_map: Dictionary mapping item_id -> feature_vector
        top_k: Number of recommendations to return
        exclude_interacted: List of item IDs to exclude
    
    Returns:
        List of (item_id, similarity_score) tuples
    """
    if exclude_interacted:
        candidate_items = {
            item_id: features
            for item_id, features in item_feature_map.items()
            if item_id not in exclude_interacted
        }
    else:
        candidate_items = item_feature_map
    
    if not candidate_items:
        return []
    
    # Normalize user profile
    user_norm = user_profile / (np.linalg.norm(user_profile) + 1e-8)
    
    similarities = []
    
    for item_id, item_features in candidate_items.items():
        # Compute cosine similarity
        item_norm = item_features / (np.linalg.norm(item_features) + 1e-8)
        similarity = np.dot(user_norm, item_norm)
        similarities.append((item_id, float(similarity)))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def batch_predict_content(
    model: tf.keras.Model,
    user_ids: np.ndarray,
    item_features: np.ndarray
) -> np.ndarray:
    """
    Make batch predictions for multiple user-item pairs.
    
    Args:
        model: Trained content-based model
        user_ids: Array of user IDs
        item_features: Array of item feature vectors (n_samples, n_features)
    
    Returns:
        Array of predicted scores
    """
    user_tensor = tf.constant(user_ids, dtype=tf.int32)
    item_tensor = tf.constant(item_features, dtype=tf.float32)
    
    predictions = model((user_tensor, item_tensor), training=False)
    return predictions.numpy().flatten()


def recommend_for_all_users_content(
    model: tf.keras.Model,
    item_feature_map: Dict[int, np.ndarray],
    top_k: int = 10,
    user_item_interactions: Optional[pd.DataFrame] = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Generate content-based recommendations for all users.
    
    Args:
        model: Trained content-based model
        item_feature_map: Dictionary mapping item_id -> feature_vector
        top_k: Number of recommendations per user
        user_item_interactions: DataFrame with columns [user_id, item_id]
                                to exclude already interacted items
    
    Returns:
        Dictionary mapping user_id to list of (item_id, score) recommendations
    """
    all_recommendations = {}
    
    # Create exclusion map if interactions provided
    exclusion_map = {}
    if user_item_interactions is not None:
        for _, row in user_item_interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            if user_id not in exclusion_map:
                exclusion_map[user_id] = []
            exclusion_map[user_id].append(item_id)
    
    # Get all unique user IDs (assuming sequential encoding starting from 0)
    if user_item_interactions is not None:
        max_user_id = user_item_interactions['user_id'].max()
        user_ids = list(range(int(max_user_id) + 1))
    else:
        # If no interactions provided, we can't determine user count
        # This would need to be passed as a parameter in practice
        raise ValueError("user_item_interactions must be provided to determine user IDs")
    
    for user_id in user_ids:
        exclude_items = exclusion_map.get(user_id, [])
        recommendations = get_content_based_recommendations(
            model,
            user_id,
            item_feature_map,
            top_k=top_k,
            exclude_interacted=exclude_items
        )
        all_recommendations[user_id] = recommendations
    
    return all_recommendations


def explain_recommendation(
    item_id: int,
    item_feature_map: Dict[int, np.ndarray],
    feature_names: List[str],
    user_profile: Optional[np.ndarray] = None
) -> Dict:
    """
    Explain why an item was recommended based on its features.
    
    Args:
        item_id: ID of the recommended item
        item_feature_map: Dictionary mapping item_id -> feature_vector
        feature_names: List of feature names corresponding to feature vector
        user_profile: Optional user preference vector for comparison
    
    Returns:
        Dictionary with explanation details
    """
    if item_id not in item_feature_map:
        raise ValueError(f"Item {item_id} not found in feature map")
    
    item_features = item_feature_map[item_id]
    
    # Get active features (non-zero)
    active_features = [
        (feature_names[i], float(item_features[i]))
        for i in range(len(feature_names))
        if item_features[i] > 0
    ]
    
    explanation = {
        'item_id': item_id,
        'active_features': active_features,
        'feature_vector': item_features.tolist()
    }
    
    if user_profile is not None:
        # Compute feature-level similarity
        feature_similarities = []
        for i, feature_name in enumerate(feature_names):
            if item_features[i] > 0 and user_profile[i] > 0:
                similarity = min(item_features[i], user_profile[i])
                feature_similarities.append((feature_name, float(similarity)))
        
        feature_similarities.sort(key=lambda x: x[1], reverse=True)
        explanation['matching_features'] = feature_similarities
    
    return explanation

