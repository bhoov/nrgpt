# TODO:
# - in create_tree, replace input_mod with a list of inputs

# we will make a simple rooted tree to represent the computation graph of nested functions

# we will use the following structure:
#   - a node is a function
#   - a leaf is a variable
#   - a root is a function that takes the input variables and returns the output variables
#   - a path from the root to a leaf is a function that takes the input variables and returns the output variable

from typing import Any
import random
from .tokenizer import COT_TOKEN, SEP, TOKEN_EOS, TOKENS_OTHER, SHARED_TOKENS


class Node:
    def __init__(self, name, func=None, value=None, max_children=None):
        """# Node class
        We want to create nested functions. 
        We can make these using a rooted tree. 
        1. Root node
        2. Each "function" node has: 
            1. an operation 
            2. a list of child nodes
        3. The "leaf" nodes must be input variables.

        Args:
            name (str): _description_
            func (callable, optional): function to act on inputs (children). Only for non-leaf nodes. Defaults to None.
            value (Any, optional): Only for leaf nodes. Defaults to None.
            max_children (int, optional): For when the node should have a limited number of children 
                (e.g., `f=lambda x1,x2: x1-x2` needs two inputs). Defaults to None.
        """
        self.name = name
        if value is not None and func is not None:
            raise ValueError("Leaf nodes cannot have a function.")
        if func is not None and value is not None:
            raise ValueError("Non-leaf nodes cannot have a value.")
        if func is None and value is None:
            raise ValueError("Nodes must have either a function or a value.")
        if func is not None:
            if not callable(func):
                raise ValueError("Function must be callable.")
        self.func = func
        self.value = value
        self.max_children = max_children
        if max_children is not None and max_children < 0:
            raise ValueError("max_children must be a positive integer.")
        self.children = []
        self.parents = []

    def add_child(self, child):
        if self.max_children is not None and len(self.children) >= self.max_children:
            raise ValueError(f"Node {self.name} has reached its maximum number of children ({self.max_children}).")
        if not isinstance(child, Node):
            raise ValueError("Child must be a Node.")
        self.children.append(child)
        child.parents.append(self)
        
        
    def add_children(self, *children):
        for child in children:
            self.add_child(child)

    # assuming the structure is a tree, we can make the compute function return its value
    def compute(self, verbose=False):
        """Compute the value of the node by applying the function to its children.
        If the node is a leaf, it returns its value.
        If the node is a function, it computes the value by applying the function to its children.
        If the node has no children, it returns its value (if it is a leaf) or None (if it is a function).
        """
        if self.func is not None:
            inputs = [child.compute(verbose) for child in self.children]
            if verbose: print(f"Computing {self.name}, func {self.func.__name__} with inputs {inputs}", end="=> ")
            self.value = self.func(*inputs)
            if verbose: print(f"Result: {self.value}")
        return self.value
    
    def __repr__(self):
        if self.func is not None:
            return f"Node({self.name}, func={self.func.__name__})"
        else:
            return f"Node({self.name}, value={self.value})"
        
    def __str__(self):
        # use this represent the whole string of the expression
        if self.func is not None:
            return f"{self.func.__name__}({','.join([str(child) for child in self.children])})"
        else:
            return str(self.value)
        
    def tokenize(self, sep = SEP):
        if self.func is not None:
            return f"{self.func.__name__}{sep}({sep}{sep.join([child.tokenize() for child in self.children])}{sep})"
        else:
            return str(self.value)
        
    def __getitem__(self, index):
        # this is a bit tricky, we need to represent the function and its inputs
        if self.func is not None:
            return self.children[index]
        else:
            raise ValueError("Leaf nodes do not have children.")
        
    def simplify(self, verbose=False):
        """Simplify the node by computing its value and removing its children."""
        if self.func is not None:
            self.compute(verbose)
        self.func = None  # Remove the function
        self.children = []  # Remove all children
        if verbose: print(f"Simplified {self.name} to value {self.value}")
            
    def get_next_func_node(node):
        """Get the next function node in the tree."""
        if node.func is not None:
            # If this node is a function, we will check its children
            for child in node.children:
                if child.func is not None:
                    return child
        return None

    def get_deepest_func_node(self, max_tries=100):
        """Get the deepest function node in the tree."""
        # while (node is not None):
        node = self  # Start from the current node
        for _ in range(max_tries):
            next_node = node.get_next_func_node()
            if next_node is None:
                return node
            node = next_node
            
    def get_CoT(root, max_iters=100, verbose=False):
        """Get the Chain of Thought (CoT) for the given root node."""
        cot = []
        for _ in range(max_iters):
            node = root.get_deepest_func_node() 
            cot.append(root.tokenize(sep=SEP))
            if verbose: print(node)
            if node.func is None:
                if verbose: print("No more function nodes to simplify.")
                break
            node.simplify(verbose=verbose)
            
        if verbose:
            print("All steps:")
            for step in cot:
                print(step)
                
        return (SEP + COT_TOKEN + SEP).join(cot) + SEP + TOKEN_EOS
                


# create a tree with random functions and values
# let's simplify this.
# we don't need to modify the current node.
# we can just create the children and add them to the current node.
def create_tree(
    node,
    depth=0,
    max_depth=3,
    min_children=1,
    max_children=3,
    input_set=list(range(10)),
    func_node_prob=0.5,
    funcs=[min,max],
    verbose=False,
    # node_class=Node,
):
    """Recursively create a tree structure with nodes and leaves.
    Args:
        node (Node): The current node to add children to.
        depth (int, optional): Current depth in the tree. Defaults to 0.
        max_depth (int, optional): Maximum depth of the tree. Defaults to 3.
        max_children (int, optional): Maximum number of children per node. Defaults to 3.
        input_mod (int, optional): Modulus for the values of leaf nodes. Defaults to 10.
        func_node_prob (float, optional): Probability of a node being a function node. Defaults to 0.5.
    """
    NodeClass = type(node)  # get the class of the node
    if (depth < max_depth): # and (node.max_children is None or len(node.children) < node.max_children):
        # randomly decide if the node is a function or a leaf
        num_children = random.randint(min_children, max_children)
        for _ in range(num_children):
            name_prefix = f"{node.name}_{len(node.children)}"
            if (random.random() < func_node_prob) and (depth < max_depth - 1):
                # create a function node
                func = random.choice(funcs)     
                name = f"{name_prefix}_f:{func.__name__}"
                if verbose: print(f"Creating node {name} at depth {depth + 1}")
                child = NodeClass(name=name, func=func)
                create_tree(
                    node=child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    min_children=min_children,
                    max_children=max_children,
                    input_set=input_set,
                    func_node_prob=func_node_prob,
                    funcs=funcs,
                    verbose=verbose,
                    # node_class=node_class,
                )
            else:
                # create a leaf node
                # num = random.randint(0, input_mod - 1)
                num = random.choice(input_set)
                name = f"{name_prefix}_x:{num}"
                # create a leaf node with a random value
                child = NodeClass(name=name, value=num)
            node.add_child(child)
    