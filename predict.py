"""
Prediction and recommendation utilities for collaborative filtering models.

This module provides functions for making predictions and generating
recommendations using trained collaborative filtering models.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import pandas as pd


def predict_user_item_rating(
    model: tf.keras.Model,
    user_id: int,
    item_id: int
) -> float:
    """
    Predict rating for a specific user-item pair.
    
    Args:
        model: Trained collaborative filtering model
        user_id: Encoded user ID
        item_id: Encoded item ID
    
    Returns:
        Predicted rating score (0-1 for binary, or raw score)
    """
    user_tensor = tf.constant([[user_id]], dtype=tf.int32)
    item_tensor = tf.constant([[item_id]], dtype=tf.int32)
    
    prediction = model((user_tensor, item_tensor), training=False)
    return float(prediction.numpy()[0][0])


def get_user_recommendations(
    model: tf.keras.Model,
    user_id: int,
    item_ids: List[int],
    top_k: int = 10,
    exclude_interacted: Optional[List[int]] = None
) -> List[Tuple[int, float]]:
    """
    Get top-k item recommendations for a user.
    
    Args:
        model: Trained collaborative filtering model
        user_id: Encoded user ID
        item_ids: List of candidate item IDs to consider
        top_k: Number of top recommendations to return
        exclude_interacted: List of item IDs to exclude (already interacted)
    
    Returns:
        List of (item_id, score) tuples, sorted by score descending
    """
    if exclude_interacted:
        item_ids = [item_id for item_id in item_ids if item_id not in exclude_interacted]
    
    if not item_ids:
        return []
    
    # Batch predictions for efficiency
    user_ids = np.array([user_id] * len(item_ids))
    item_ids_array = np.array(item_ids)
    
    user_tensor = tf.constant(user_ids, dtype=tf.int32)
    item_tensor = tf.constant(item_ids_array, dtype=tf.int32)
    
    predictions = model((user_tensor, item_tensor), training=False)
    scores = predictions.numpy().flatten()
    
    # Get top-k items
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    recommendations = [
        (item_ids[idx], float(scores[idx]))
        for idx in top_indices
    ]
    
    return recommendations


def get_item_similar_items(
    model: tf.keras.Model,
    item_id: int,
    all_item_ids: List[int],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find items similar to a given item using item embeddings.
    
    Args:
        model: Trained collaborative filtering model
        item_id: Encoded item ID to find similar items for
        all_item_ids: List of all candidate item IDs
        top_k: Number of similar items to return
    
    Returns:
        List of (item_id, similarity_score) tuples
    """
    # Get item embedding
    item_embedding_layer = model.get_layer('item_embedding')
    item_emb = item_embedding_layer(tf.constant([[item_id]], dtype=tf.int32))
    item_emb = tf.reshape(item_emb, [-1])
    
    # Get embeddings for all items
    all_item_tensor = tf.constant(all_item_ids, dtype=tf.int32)
    all_item_embs = item_embedding_layer(all_item_tensor)
    
    # Compute cosine similarity
    item_emb_norm = tf.nn.l2_normalize(item_emb, axis=0)
    all_item_embs_norm = tf.nn.l2_normalize(all_item_embs, axis=1)
    
    similarities = tf.reduce_sum(
        item_emb_norm * all_item_embs_norm,
        axis=1
    ).numpy()
    
    # Get top-k similar items (excluding the item itself)
    similar_items = [
        (all_item_ids[idx], float(similarities[idx]))
        for idx in np.argsort(similarities)[::-1]
        if all_item_ids[idx] != item_id
    ][:top_k]
    
    return similar_items


def batch_predict(
    model: tf.keras.Model,
    user_ids: np.ndarray,
    item_ids: np.ndarray
) -> np.ndarray:
    """
    Make batch predictions for multiple user-item pairs.
    
    Args:
        model: Trained collaborative filtering model
        user_ids: Array of user IDs
        item_ids: Array of item IDs
    
    Returns:
        Array of predicted scores
    """
    user_tensor = tf.constant(user_ids, dtype=tf.int32)
    item_tensor = tf.constant(item_ids, dtype=tf.int32)
    
    predictions = model((user_tensor, item_tensor), training=False)
    return predictions.numpy().flatten()


def recommend_for_all_users(
    model: tf.keras.Model,
    num_users: int,
    num_items: int,
    top_k: int = 10,
    user_item_interactions: Optional[pd.DataFrame] = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Generate recommendations for all users.
    
    Args:
        model: Trained collaborative filtering model
        num_users: Total number of users
        num_items: Total number of items
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
    
    all_item_ids = list(range(num_items))
    
    for user_id in range(num_users):
        exclude_items = exclusion_map.get(user_id, [])
        recommendations = get_user_recommendations(
            model,
            user_id,
            all_item_ids,
            top_k=top_k,
            exclude_interacted=exclude_items
        )
        all_recommendations[user_id] = recommendations
    
    return all_recommendations


def decode_recommendations(
    recommendations: List[Tuple[int, float]],
    item_mapping: Dict[int, int]
) -> List[Tuple[int, float]]:
    """
    Decode encoded item IDs back to original IDs using mapping.
    
    Args:
        recommendations: List of (encoded_item_id, score) tuples
        item_mapping: Dictionary mapping encoded_id -> original_id
    
    Returns:
        List of (original_item_id, score) tuples
    """
    reverse_mapping = {v: k for k, v in item_mapping.items()}
    
    decoded = [
        (reverse_mapping.get(item_id, item_id), score)
        for item_id, score in recommendations
    ]
    
    return decoded

