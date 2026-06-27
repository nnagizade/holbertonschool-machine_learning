#!/usr/bin/env python3
"""Decision Tree module with Gini split criterion"""


import numpy as np


class Node:
    """Node class for Decision Tree"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize Node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return max depth below this node"""
        if self.is_leaf:
            return self.depth
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node"""
        if self.is_leaf:
            return 1
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Add prefix for left child"""
        lines = text.split("\n")
        new_text = "+---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("| " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child"""
        lines = text.split("\n")
        new_text = "+---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("  " + x) + "\n"
        return new_text

    def __str__(self):
        """Return string representation of Node"""
        if self.is_root:
            title = "root"
        else:
            title = "node"
        node_str = (
            f"{title} [feature={self.feature},"
            f" threshold={self.threshold}]\n"
        )
        left_str = self.left_child_add_prefix(self.left_child.__str__())
        right_str = self.right_child_add_prefix(self.right_child.__str__())
        return node_str + left_str + right_str

    def get_leaves_below(self):
        """Return list of leaves below this node"""
        if self.is_leaf:
            return [self]
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Update bounds for all nodes below"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child, direction in [
            (self.left_child, "left"),
            (self.right_child, "right")
        ]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
            if direction == "left":
                child.upper[self.feature] = min(
                    child.upper.get(self.feature, np.inf),
                    self.threshold
                )
            else:
                child.lower[self.feature] = max(
                    child.lower.get(self.feature, -np.inf),
                    self.threshold
                )

        if not self.left_child.is_leaf:
            self.left_child.update_bounds_below()
        if not self.right_child.is_leaf:
            self.right_child.update_bounds_below()

    def update_indicator(self):
        """Update indicator function using bounds"""
        def is_large_enough(x):
            return np.all(
                np.array([x[:, key] > self.lower[key]
                          for key in self.lower]), axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([x[:, key] <= self.upper[key]
                          for key in self.upper]), axis=0
            )

        self.indicator = lambda x: (
            np.logical_and(is_large_enough(x), is_small_enough(x))
        )

    def pred(self, x):
        """Return prediction for input x"""
        if x[self.feature] > self.threshold:
            return self.right_child.pred(x)
        return self.left_child.pred(x)


class Leaf(Node):
    """Leaf class for Decision Tree"""

    def __init__(self, value, depth=None):
        """Initialize Leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return depth of this leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 for leaf node"""
        return 1

    def __str__(self):
        """Return string representation of Leaf"""
        return f"leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return self as list"""
        return [self]

    def update_bounds_below(self):
        """No bounds to update for leaf"""
        pass

    def pred(self, x):
        """Return leaf value as prediction"""
        return self.value


class Decision_Tree:
    """Decision Tree class"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize Decision Tree"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return max depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return count of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return string representation of Decision Tree"""
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Return list of all leaves"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Update prediction function using leaves"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([
            leaves[np.argmax(
                np.array([leaf.indicator(A) for leaf in leaves]),
                axis=0
            )[i]].value
            for i in range(A.shape[0])
        ])

    def pred(self, x):
        """Return prediction for single input"""
        return self.root.pred(x)

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy on test data"""
        return np.sum(
            np.array([self.pred(test_explanatory[i])
                      for i in range(test_explanatory.shape[0])]) ==
            test_target
        ) / test_target.size

    def possible_thresholds(self, node, feature):
        """Return possible thresholds for a feature"""
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return best threshold and Gini for one feature"""
        thresholds = self.possible_thresholds(node, feature)
        n = node.sub_population.sum()

        classes = np.unique(self.target)
        c = len(classes)
        t = len(thresholds)

        # Build Left_F array of shape (n_individuals, t, c)
        individuals = self.explanatory[node.sub_population, feature]
        targets = self.target[node.sub_population]

        # Shape (n, t): True if individual i goes left for threshold j
        left_mask = individuals[:, np.newaxis] <= thresholds[np.newaxis, :]

        # Shape (n, c): one-hot class encoding
        class_mask = (
            targets[:, np.newaxis] == classes[np.newaxis, :]
        )

        # Left child Gini impurities
        left_counts = np.einsum('it,ic->tc', left_mask, class_mask)
        left_sizes = left_mask.sum(axis=0)  # shape (t,)
        left_sizes_safe = np.where(left_sizes == 0, 1, left_sizes)
        left_probs = left_counts / left_sizes_safe[:, np.newaxis]
        left_gini = 1 - np.sum(left_probs ** 2, axis=1)  # shape (t,)

        # Right child Gini impurities
        right_mask = ~left_mask
        right_counts = np.einsum('it,ic->tc', right_mask, class_mask)
        right_sizes = right_mask.sum(axis=0)  # shape (t,)
        right_sizes_safe = np.where(right_sizes == 0, 1, right_sizes)
        right_probs = right_counts / right_sizes_safe[:, np.newaxis]
        right_gini = 1 - np.sum(right_probs ** 2, axis=1)  # shape (t,)

        # Weighted average Gini
        gini_split = (
            left_sizes / n * left_gini +
            right_sizes / n * right_gini
        )

        best_idx = np.argmin(gini_split)
        return thresholds[best_idx], gini_split[best_idx]

    def Gini_split_criterion(self, node):
        """Return best feature and threshold using Gini criterion"""
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def random_split_criterion(self, node):
        """Return random feature and threshold for splitting"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            thresholds = self.possible_thresholds(node, feature)
            if len(thresholds) > 0:
                threshold = self.rng.choice(thresholds)
                left = self.explanatory[
                    node.sub_population & (
                        self.explanatory[:, feature] > threshold
                    )
                ]
                right = self.explanatory[
                    node.sub_population & (
                        self.explanatory[:, feature] <= threshold
                    )
                ]
                diff = len(left) - len(right)
        return feature, threshold

    def get_leaf_child(self, node, direction):
        """Create a leaf child node"""
        if direction == "left":
            sub_pop = (
                node.sub_population &
                (self.explanatory[:, node.feature] <= node.threshold)
            )
        else:
            sub_pop = (
                node.sub_population &
                (self.explanatory[:, node.feature] > node.threshold)
            )
        value = np.argmax(
            np.bincount(self.target[sub_pop])
        )
        leaf = Leaf(value, depth=node.depth + 1)
        leaf.sub_population = sub_pop
        return leaf

    def get_node_child(self, node, direction):
        """Create a non-leaf child node"""
        if direction == "left":
            sub_pop = (
                node.sub_population &
                (self.explanatory[:, node.feature] <= node.threshold)
            )
        else:
            sub_pop = (
                node.sub_population &
                (self.explanatory[:, node.feature] > node.threshold)
            )
        child = Node(depth=node.depth + 1)
        child.sub_population = sub_pop
        return child

    def fit_node(self, node):
        """Fit a node recursively"""
        if self.split_criterion == "random":
            node.feature, node.threshold = self.random_split_criterion(node)
        else:
            node.feature, node.threshold = self.Gini_split_criterion(node)

        left_pop = (
            node.sub_population &
            (self.explanatory[:, node.feature] <= node.threshold)
        )
        right_pop = (
            node.sub_population &
            (self.explanatory[:, node.feature] > node.threshold)
        )

        is_left_leaf = (
            node.depth + 1 >= self.max_depth or
            left_pop.sum() < self.min_pop or
            len(np.unique(self.target[left_pop])) == 1
        )
        is_right_leaf = (
            node.depth + 1 >= self.max_depth or
            right_pop.sum() < self.min_pop or
            len(np.unique(self.target[right_pop])) == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, "left")
        else:
            node.left_child = self.get_node_child(node, "left")
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, "right")
        else:
            node.right_child = self.get_node_child(node, "right")
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree to training data"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones(target.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(explanatory, target)}""")

    def update_predict(self):
        """Update prediction using leaf indicators"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([
            leaves[np.argmax(
                np.array([leaf.indicator(A) for leaf in leaves]),
                axis=0
            )[i]].value
            for i in range(A.shape[0])
        ])

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy on test data"""
        return np.sum(
            np.array([self.pred(test_explanatory[i])
                      for i in range(test_explanatory.shape[0])]) ==
            test_target
        ) / test_target.size
