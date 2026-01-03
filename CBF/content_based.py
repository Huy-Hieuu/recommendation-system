"""
Content-Based Recommendation Model using TensorFlow.

This module implements deep learning models that learn user preferences
based on item content features (e.g., genres, descriptions, metadata).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, List


class ContentBasedModel(keras.Model):
    """
    Content-based recommendation model using item features.
    
    Learns user preferences by analyzing item content features and
    user interaction patterns. Uses deep neural networks to capture
    complex feature interactions.
    """
    
    def __init__(
        self,
        num_users: int,
        item_feature_dim: int,
        user_embedding_dim: int = 50,
        feature_hidden_layers: List[int] = None,
        combined_hidden_layers: List[int] = None,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the content-based model.
        
        Args:
            num_users: Total number of unique users
            item_feature_dim: Dimension of item feature vector
            user_embedding_dim: Dimension of user embedding
            feature_hidden_layers: Hidden layer sizes for feature processing
            combined_hidden_layers: Hidden layer sizes after combining user/item
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional arguments passed to keras.Model
        """
        super().__init__(**kwargs)
        
        if feature_hidden_layers is None:
            feature_hidden_layers = [128, 64]
        if combined_hidden_layers is None:
            combined_hidden_layers = [64, 32, 16]
        
        self.num_users = num_users
        self.item_feature_dim = item_feature_dim
        self.user_embedding_dim = user_embedding_dim
        
        # User embedding layer
        self.user_embedding = layers.Embedding(
            num_users,
            user_embedding_dim,
            name='user_embedding'
        )
        self.user_flatten = layers.Flatten()
        
        # Item feature processing layers
        self.feature_layers = []
        self.feature_dropout_layers = []
        
        for hidden_size in feature_hidden_layers:
            self.feature_layers.append(
                layers.Dense(hidden_size, activation='relu')
            )
            self.feature_dropout_layers.append(
                layers.Dropout(dropout_rate)
            )
        
        # Concatenation layer
        self.concat = layers.Concatenate()
        
        # Combined hidden layers
        self.combined_layers = []
        self.combined_dropout_layers = []
        
        for hidden_size in combined_hidden_layers:
            self.combined_layers.append(
                layers.Dense(hidden_size, activation='relu')
            )
            self.combined_dropout_layers.append(
                layers.Dropout(dropout_rate)
            )
        
        # Output layer
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Tuple of (user_ids, item_features) tensors
            training: Whether the model is in training mode
        
        Returns:
            Prediction scores for user-item pairs
        """
        user_ids, item_features = inputs
        
        # Get user embedding
        user_emb = self.user_embedding(user_ids)
        user_emb = self.user_flatten(user_emb)
        
        # Process item features
        x_features = item_features
        for feature_layer, dropout_layer in zip(
            self.feature_layers, self.feature_dropout_layers
        ):
            x_features = feature_layer(x_features)
            x_features = dropout_layer(x_features, training=training)
        
        # Concatenate user embedding and processed item features
        combined = self.concat([user_emb, x_features])
        
        # Pass through combined layers
        x = combined
        for combined_layer, dropout_layer in zip(
            self.combined_layers, self.combined_dropout_layers
        ):
            x = combined_layer(x)
            x = dropout_layer(x, training=training)
        
        # Output prediction
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'item_feature_dim': self.item_feature_dim,
            'user_embedding_dim': self.user_embedding_dim,
        })
        return config


class SimpleContentBasedModel(keras.Model):
    """
    Simple content-based model using cosine similarity approach.
    
    Learns user preference vectors and item feature vectors,
    then computes similarity for recommendations.
    """
    
    def __init__(
        self,
        num_users: int,
        item_feature_dim: int,
        embedding_dim: int = 50,
        **kwargs
    ):
        """
        Initialize the simple content-based model.
        
        Args:
            num_users: Total number of unique users
            item_feature_dim: Dimension of item feature vector
            embedding_dim: Dimension of user preference embedding
            **kwargs: Additional arguments passed to keras.Model
        """
        super().__init__(**kwargs)
        
        self.num_users = num_users
        self.item_feature_dim = item_feature_dim
        self.embedding_dim = embedding_dim
        
        # User preference embedding (learned from interactions)
        self.user_preference = layers.Embedding(
            num_users,
            embedding_dim,
            name='user_preference'
        )
        self.user_flatten = layers.Flatten()
        
        # Item feature projection to same dimension
        self.item_projection = layers.Dense(
            embedding_dim,
            activation='linear',
            name='item_projection'
        )
        
        # Dot product for similarity
        self.dot_product = layers.Dot(axes=1)
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Tuple of (user_ids, item_features) tensors
            training: Whether the model is in training mode
        
        Returns:
            Similarity scores for user-item pairs
        """
        user_ids, item_features = inputs
        
        # Get user preference vector
        user_pref = self.user_preference(user_ids)
        user_pref = self.user_flatten(user_pref)
        
        # Project item features to same dimension
        item_proj = self.item_projection(item_features)
        
        # Compute dot product (similarity)
        output = self.dot_product([user_pref, item_proj])
        
        return output
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'item_feature_dim': self.item_feature_dim,
            'embedding_dim': self.embedding_dim,
        })
        return config


def create_content_based_model(
    num_users: int,
    item_feature_dim: int,
    user_embedding_dim: int = 50,
    feature_hidden_layers: List[int] = None,
    combined_hidden_layers: List[int] = None,
    dropout_rate: float = 0.2
) -> ContentBasedModel:
    """
    Factory function to create a content-based recommendation model.
    
    Args:
        num_users: Total number of unique users
        item_feature_dim: Dimension of item feature vector
        user_embedding_dim: Dimension of user embedding
        feature_hidden_layers: Hidden layer sizes for feature processing
        combined_hidden_layers: Hidden layer sizes after combining user/item
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Content-based model ready for training
    """
    model = ContentBasedModel(
        num_users=num_users,
        item_feature_dim=item_feature_dim,
        user_embedding_dim=user_embedding_dim,
        feature_hidden_layers=feature_hidden_layers,
        combined_hidden_layers=combined_hidden_layers,
        dropout_rate=dropout_rate
    )
    
    return model


def create_simple_content_model(
    num_users: int,
    item_feature_dim: int,
    embedding_dim: int = 50
) -> SimpleContentBasedModel:
    """
    Factory function to create a simple content-based model.
    
    Args:
        num_users: Total number of unique users
        item_feature_dim: Dimension of item feature vector
        embedding_dim: Dimension of embeddings
    
    Returns:
        Simple content-based model ready for training
    """
    model = SimpleContentBasedModel(
        num_users=num_users,
        item_feature_dim=item_feature_dim,
        embedding_dim=embedding_dim
    )
    
    return model


def compile_content_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: Optional[keras.optimizers.Optimizer] = None
) -> keras.Model:
    """
    Compile the content-based model with appropriate loss and optimizer.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Custom optimizer (if None, uses Adam)
    
    Returns:
        Compiled model
    """
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

