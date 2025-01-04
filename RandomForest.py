import numpy as np
from collections import Counter
import random
import pandas as pd

# Decision Tree
class decisiontreebaru:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fitbaru1(self, X, y):
        self.tree = self._growtreebaru(X, y)

    def _growtreebaru(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        feature_idx, threshold = self._bestsplitbaru(X, y, num_features)
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self._growtreebaru(X[left_idx], y[left_idx], depth + 1),
            'right': self._growtreebaru(X[right_idx], y[right_idx], depth + 1)
        }

    def _bestsplitbaru(self, X, y, num_features):
        best_gini = 1.0
        best_idx = None
        best_threshold = None
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gini = self._giniindexbaru(X[:, feature_idx], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_idx
                    best_threshold = threshold
        return best_idx, best_threshold

    def _giniindexbaru(self, feature_column, y, threshold):
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold
        gini_left = 1.0 - sum((np.sum(left_idx & (y == c)) / np.sum(left_idx)) ** 2 for c in set(y))
        gini_right = 1.0 - sum((np.sum(right_idx & (y == c)) / np.sum(right_idx)) ** 2 for c in set(y))
        return (np.sum(left_idx) * gini_left + np.sum(right_idx) * gini_right) / len(y)

    def predictbaru1(self, X):
        return np.array([self._predictonebaru(x, self.tree) for x in X])

    def _predictonebaru(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature_val = x[tree['feature_idx']]
        if feature_val <= tree['threshold']:
            return self._predictonebaru(x, tree['left'])
        else:
            return self._predictonebaru(x, tree['right'])

# Random Forest
class random_forest_baru:
    def __init__(self, num_trees=10, max_depth=None, sample_size=None, class_weight=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.class_weight = class_weight
        self.trees = []

    def fitbaru(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            idxs = np.random.choice(len(y), self.sample_size, replace=True)
            tree = decisiontreebaru(max_depth=self.max_depth)

            if self.class_weight:
                sample_weight = np.array([self.class_weight[label] for label in y[idxs]])
                idxs = np.random.choice(idxs, len(idxs), p=sample_weight/sample_weight.sum())

            tree.fitbaru1(X[idxs], y[idxs])
            self.trees.append(tree)

    def predictbaru(self, X):
        tree_preds = np.array([tree.predictbaru1(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(len(X))])

    def predict_probabaru(self, X):
        tree_preds = np.array([tree.predictbaru1(X) for tree in self.trees])
        probs = []
        for i in range(len(X)):
            counts = Counter(tree_preds[:, i])
            total = sum(counts.values())
            probs.append([counts.get(c, 0) / total for c in range(3)])
        return np.array(probs)