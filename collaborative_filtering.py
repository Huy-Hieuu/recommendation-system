"""
Neural Collaborative Filtering (NCF) Model for Recommendation Systems.

This module implements a deep learning-based collaborative filtering approach
using TensorFlow/Keras. The model learns user and item embeddings through
matrix factorization with neural networks.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class NeuralCollaborativeFiltering(keras.Model):
    """
    Neural Collaborative Filtering model for recommendation systems.
    
    Combines matrix factorization with neural networks to learn user-item
    interactions. Uses embedding layers for users and items, followed by
    deep neural network layers to capture non-linear patterns.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 50,
        hidden_layers: list = None,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the NCF model.
        
        Args:
            num_users: Total number of unique users in the dataset
            num_items: Total number of unique items in the dataset
            embedding_dim: Dimension of user and item embeddings
            hidden_layers: List of hidden layer sizes (default: [64, 32, 16])
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional arguments passed to keras.Model
        """
        super().__init__(**kwargs)
        
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and item embedding layers
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_dim,
            name='user_embedding'
        )
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_dim,
            name='item_embedding'
        )
        
        # Flatten layers for embeddings
        self.user_flatten = layers.Flatten()
        self.item_flatten = layers.Flatten()
        
        # Concatenation layer
        self.concat = layers.Concatenate()
        
        # Hidden layers with dropout
        self.hidden_layers = []
        self.dropout_layers = []
        
        for hidden_size in hidden_layers:
            self.hidden_layers.append(
                layers.Dense(hidden_size, activation='relu')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate)
            )
        
        # Output layer (binary classification: like/dislike or rating prediction)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Tuple of (user_ids, item_ids) tensors
            training: Whether the model is in training mode
        
        Returns:
            Prediction scores for user-item pairs
        """
        user_ids, item_ids = inputs
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Flatten embeddings
        user_emb = self.user_flatten(user_emb)
        item_emb = self.item_flatten(item_emb)
        
        # Concatenate user and item embeddings
        concat_emb = self.concat([user_emb, item_emb])
        
        # Pass through hidden layers
        x = concat_emb
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)
        
        # Output prediction
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
        })
        return config


class MatrixFactorizationModel(keras.Model):
    """
    Simple Matrix Factorization model using dot product of embeddings.
    
    A lightweight alternative to NCF that uses only embedding layers
    and dot product for predictions. Faster and more interpretable.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 50,
        **kwargs
    ):
        """
        Initialize the Matrix Factorization model.
        
        Args:
            num_users: Total number of unique users in the dataset
            num_items: Total number of unique items in the dataset
            embedding_dim: Dimension of user and item embeddings
            **kwargs: Additional arguments passed to keras.Model
        """
        super().__init__(**kwargs)
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and item embedding layers
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_dim,
            name='user_embedding'
        )
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_dim,
            name='item_embedding'
        )
        
        # Dot product layer
        self.dot_product = layers.Dot(axes=1)
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Tuple of (user_ids, item_ids) tensors
            training: Whether the model is in training mode (unused for this model)
        
        Returns:
            Dot product scores for user-item pairs
        """
        user_ids, item_ids = inputs
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Compute dot product
        output = self.dot_product([user_emb, item_emb])
        
        return output
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
        })
        return config


def create_ncf_model(
    num_users: int,
    num_items: int,
    embedding_dim: int = 50,
    hidden_layers: list = None,
    dropout_rate: float = 0.2
) -> NeuralCollaborativeFiltering:
    """
    Factory function to create a Neural Collaborative Filtering model.
    
    Args:
        num_users: Total number of unique users
        num_items: Total number of unique items
        embedding_dim: Dimension of embeddings
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled NCF model ready for training
    """
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate
    )
    
    return model


def create_mf_model(
    num_users: int,
    num_items: int,
    embedding_dim: int = 50
) -> MatrixFactorizationModel:
    """
    Factory function to create a Matrix Factorization model.
    
    Args:
        num_users: Total number of unique users
        num_items: Total number of unique items
        embedding_dim: Dimension of embeddings
    
    Returns:
        Compiled MF model ready for training
    """
    model = MatrixFactorizationModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim
    )
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: Optional[keras.optimizers.Optimizer] = None
) -> keras.Model:
    """
    Compile the model with appropriate loss and optimizer.
    
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

