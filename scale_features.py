import numpy as np


def scale_features(features: np.array) -> np.array:
    features_num = len(features[0])
    samples_num = len(features)

    feature_ranges = []

    for feature_idx in range(features_num):
        min_feature_range = np.infty
        max_feature_range = -np.infty
        for features_sample_idx in range(samples_num):
            feature_value = features[features_sample_idx][feature_idx]
            min_feature_range = min(min_feature_range, feature_value)
            max_feature_range = max(max_feature_range, feature_value)

        feature_ranges.append((min_feature_range, max_feature_range))

    for features_sample_idx in range(samples_num):
        for feature_idx in range(features_num):
            original_feature_val = features[features_sample_idx][feature_idx]
            min_feature_range = feature_ranges[feature_idx][0]
            max_feature_range = feature_ranges[feature_idx][1]
            features[features_sample_idx][feature_idx] = \
                (original_feature_val - min_feature_range) / (max_feature_range - min_feature_range)

    return features
