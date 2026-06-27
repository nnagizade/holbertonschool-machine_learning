#!/usr/bin/env python3
"""Decision Tree module"""


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
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False

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


class Leaf:
    """Leaf class for Decision Tree"""

    def __init__(self, value, depth=None):
        """Initialize Leaf"""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Return string representation of Leaf"""
        return f"leaf [value={self.value}]"


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

    def __str__(self):
        """Return string representation of Decision Tree"""
        return self.root.__str__()
