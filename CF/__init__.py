"""
Collaborative Filtering (CF) Recommendation System.

This package contains implementations of collaborative filtering algorithms
including Neural Collaborative Filtering (NCF) and Matrix Factorization.
"""

from .collaborative_filtering import (
    NeuralCollaborativeFiltering,
    MatrixFactorizationModel,
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

__all__ = [
    'NeuralCollaborativeFiltering',
    'MatrixFactorizationModel',
    'create_ncf_model',
    'create_mf_model',
    'compile_model',
    'load_interaction_data',
    'encode_user_item_ids',
    'create_rating_column',
    'prepare_training_data',
    'split_train_test',
    'create_tf_dataset'
]

