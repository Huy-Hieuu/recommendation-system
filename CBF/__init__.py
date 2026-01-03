"""
Content-Based Filtering (CBF) Recommendation System.

This package contains implementations of content-based recommendation algorithms
that use item features (e.g., genres, metadata) to make recommendations.
"""

from .content_based import (
    ContentBasedModel,
    SimpleContentBasedModel,
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
    create_content_tf_dataset,
    build_user_profile
)

__all__ = [
    'ContentBasedModel',
    'SimpleContentBasedModel',
    'create_content_based_model',
    'create_simple_content_model',
    'compile_content_model',
    'load_movielens_items',
    'extract_genre_features',
    'normalize_features',
    'create_item_feature_map',
    'prepare_content_based_data',
    'generate_negative_samples_content',
    'split_content_data',
    'create_content_tf_dataset',
    'build_user_profile'
]

