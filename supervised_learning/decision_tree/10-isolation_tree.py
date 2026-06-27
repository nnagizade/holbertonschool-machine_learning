#!/usr/bin/env python3
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
import numpy as np


class Isolation_Random_Tree():
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        def node_to_str(node, depth=0):
            indent = "    " * depth
            if isinstance(node, Leaf):
                return f"{indent}Leaf (depth={node.depth})\n"
            s = f"{indent}Node [feature={node.feature}, threshold={node.threshold}]\n"
            if node.left_child:
                s += node_to_str(node.left_child, depth + 1)
            if node.right_child:
                s += node_to_str(node.right_child, depth + 1)
            return s
        return node_to_str(self.root)

    def depth(self):
        def node_depth(node):
            if isinstance(node, Leaf):
                return node.depth
            left_depth = node_depth(node.left_child) if node.left_child else 0
            right_depth = node_depth(node.right_child) if node.right_child else 0
            return max(left_depth, right_depth)
        return node_depth(self.root)

    def count_nodes(self, only_leaves=False):
        def count(node):
            if isinstance(node, Leaf):
                return 1
            left = count(node.left_child) if node.left_child else 0
            right = count(node.right_child) if node.right_child else 0
            if only_leaves:
                return left + right
            return 1 + left + right
        return count(self.root)

    def update_bounds(self):
        def update(node, lower, upper):
            node.lower = lower.copy()
            node.upper = upper.copy()
            if isinstance(node, Leaf):
                return
            f = node.feature
            t = node.threshold
            left_upper = upper.copy()
            left_upper[f] = t
            right_lower = lower.copy()
            right_lower[f] = t
            if node.left_child:
                update(node.left_child, lower, left_upper)
            if node.right_child:
                update(node.right_child, right_lower, upper)
        n_features = self.explanatory.shape[1]
        lower = np.full(n_features, -np.inf)
        upper = np.full(n_features, np.inf)
        update(self.root, lower, upper)

    def get_leaves(self):
        leaves = []
        def collect(node):
            if isinstance(node, Leaf):
                leaves.append(node)
                return
            if node.left_child:
                collect(node.left_child)
            if node.right_child:
                collect(node.right_child)
        collect(self.root)
        return leaves

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.value = leaf.depth

        def predict_individual(x):
            node = self.root
            while not isinstance(node, Leaf):
                if x[node.feature] > node.threshold:
                    node = node.right_child
                else:
                    node = node.left_child
            return node.value

        self.predict = lambda A: np.array([predict_individual(x) for x in A])

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        threshold = self.rng.uniform(feature_min, feature_max)
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        leaf_child = Leaf(is_leaf=True)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] <= node.threshold)
        )

        # Is left node a leaf?
        is_left_leaf = (
            node.depth >= self.max_depth - 1
            or np.sum(left_population) <= self.min_pop
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (
            node.depth >= self.max_depth - 1
            or np.sum(right_population) <= self.min_pop
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0], dtype='bool'
        )
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
