from .graph import Node, create_tree
from .tokenizer import Tokenizer, TOKEN_EOS, TOKENS_OTHER, SEP

# This module defines a dataset for ListOps tasks, which generates trees of operations
# Update: 
# - Adding Polish notation option

# Dataset:
# This will contain CoT strings as a continuous stream for training.
# We want it to return a dictionary with the following contents:

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Callable, Union, Set
# import numpy as np
# import torch
import random
# random.seed(42)  # For reproducibility

# import h5py
import pickle
import time
class ListOpsDataset(BaseModel):
    """
    A dataset for ListOps tasks, containing CoT strings and their corresponding outputs.
    {
    'train': concatenated CoT strings of all samples,
    'test': [{
        'character': CoT string,
        'encoded': list of indices of the tokens in the CoT string,
        'result': list of indices of the output value in the vocabulary
    }],
    'test_single_ops': {op: [test data for single operations]},
    'metadata': {
        'vocab': list of tokens in the vocabulary,
        'max_depth': maximum depth of the trees,
        'min_children': minimum number of children for each node,
        'max_children': maximum number of children for each node,
        'func_node_prob': probability of a node being a function node,
        'all_funcs': list of all functions available,
        'funcs_to_use': list of functions actually used in the dataset,
        'input_set': list of input values to use in the trees,
        'num_train': number of training samples,
        'num_test': number of test samples,
        'exclusion_set': list of sequences that should not appear in the training dataset
    }
    }

    """
    # TOKEN_EOS: str = 'END'
    # TOKENS_REST: List[str] = list('()=')

    # num_samples: int = 100
    max_depth: int = 3
    min_children: int = 2
    max_children: int = 3
    input_set: List = Field(default=list(range(10)), description="Set of input values to use in the trees.")
    func_node_prob: float = 0.5
    all_funcs: List = Field(default=[min,max], description="List of all functions to use in the dataset. (not all may be used)")
    funcs_to_use: List = Field(default=[min], description="List of functions to actually use. (must be in all_funcs)")
    # tok2index: Optional[Dict] = None #Dict[str, int] #= Field(default_factory=dict)
    tokenizer: Optional[Tokenizer] = None
    vocab: List = Field(default_factory=list, init=False, description="Vocabulary containing function names and input values.")
    exclusion_set: Optional[Set] = None #Field(default_factory=list, description="Set of sequences that should not appear in the training dataset.")
    exclude_permutation: bool = Field(default=False, description="Whether to exclude permutations of the exclusion set elements in the training dataset.")
    data: List = Field(default_factory=list, init=False,)
    train_data: List = Field(default_factory=list, init=False, description="Concatenated CoT encoded list of int of all samples for training.")
    test_data: List = [] #Dict = Field(default_factory=dict)
    num_train: int = 0
    num_test: int = 0
    test_single_ops: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Test data for single operations, indexed by operation name.")
    polish_notation: bool = Field(default=False, description="Whether to use Polish notation for the CoT strings.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',  # Disallow extra fields
        frozen=False,  # Allow modification of the model after creation
    )

    def model_post_init(self, __context):
        """
        This method is called after the model is initialized.
        """
        assert set(self.funcs_to_use).issubset(set(self.all_funcs)), \
            f"funcs_to_use {self.funcs_to_use} must be a subset of all_funcs {self.all_funcs}"
        assert self.max_depth > 0, "max_depth must be greater than 0"
        assert self.min_children > 0, "min_children must be greater than 0"
        assert self.max_children >= self.min_children, "max_children must be greater than or equal to min_children"
        # assert self.input_set, "input_set must not be empty"
        # assert self.num_samples > 0, "num_samples must be greater than 0"
        assert 0 <= self.func_node_prob <= 1, "func_node_prob must be between 0 and 1"
        # check that input_set is subscriptable

        self.vocab = self.make_vocab(self.all_funcs, self.input_set)
        self.tokenizer = Tokenizer(self.vocab, sep=',')
        # set random seed for reproducibility
        # torch.manual_seed(self.seed)
        random.seed(self.seed)

    def make_vocab(self, all_funcs, input_set):
        """
        Create a vocabulary from the functions and input set.
        Args:
            all_funcs (list): List of all functions available.
            input_set (list): List of input values to use in the trees.
        Returns:
            str: Vocabulary containing function names and input values.
        """
        # return self.TOKEN_EOS + self.TOKENS_REST + ','.join([f.__name__ for f in all_funcs] + [str(i) for i in input_set])
        return [TOKEN_EOS] + TOKENS_OTHER + [f.__name__ for f in all_funcs] + [str(i) for i in input_set]

    def make_tree(self, name='root', max_iters=100,verbose=False, funcs=None):
        if funcs is None:
            funcs = self.funcs_to_use
        if verbose: print(self.input_set)
        root = Node(name=name, func=random.choice(funcs))
        create_tree(
            root,
            max_depth=self.max_depth,
            min_children=self.min_children,
            max_children=self.max_children,
            input_set=self.input_set,
            func_node_prob=self.func_node_prob,
            funcs=funcs,
            verbose=verbose,
        )
        return root
    
    def get_datapoint(self, name='root', verbose=False, funcs=None):
        root = self.make_tree(name=name, verbose=verbose, funcs=funcs)
        cot = root.get_CoT(max_iters=100, verbose=verbose)
        output = str(root.compute(verbose=verbose))
        if self.polish_notation:
            cot = self.to_polish_notation(cot, funcs=self.funcs_to_use)
        return cot, output
        

    def make_exclusion_set(self, sampling_rate = 0.05):# num_samples=100):
        """
        Create a set of samples that are not included in the dataset.
        This is useful for testing the model on unseen data.
        
        We design the exclusion set to be a set of sequences from the input_set that are not present in the training dataset.
        We will check this in the training loop. Each sequence looks like '1,2,3' (sep=','). 
        We will check them in the CoT strings.
        Args:
            sampling_rate (float): The rate at which to sample from the input_set to create the exclusion set.
        Returns:
            list: A list of strings of input values (comma separated) which should not appear in the training dataset. 
        """
        num_samples = int(len(self.input_set)**self.max_children * sampling_rate)
        print(f"Creating exclusion set with {num_samples} samples from input_set of size {len(self.input_set)}")
        exclusion_set = []
        # choose random input values from the input_set based on min_children and max_children
        for i in range(num_samples):
            # sampled_input = random.sample(self.input_set, random.randint(self.min_children, self.max_children))
            # only use the upper limit of max_children
            sampled_input = random.sample(self.input_set, self.max_children)
            # exclusion_set.append(SEP.join(map(str, sampled_input)))
            # if self.exclude_permutation:
            if self.exclude_permutation:
                # just sort the sampled input to avoid permutations
                sampled_input = sorted(sampled_input)
            exclusion_set.append(tuple(map(str, sampled_input)))
        return set(exclusion_set) 

    def in_exclusion_set(self, cot):
        """Checks whether a sample contains any of the sequences in the exclusion set."""
        cot_tokens = cot.split(SEP)
        # merge consecutive tokens to form sequences of len self.max_children
        # cot_seqs = set([SEP.join(cot_tokens[i:i+self.max_children]) for i in range(len(cot_tokens) - self.max_children + 1)])
        if self.exclude_permutation:
            cot_seqs = set(tuple(sorted(cot_tokens[i:i+self.max_children])) for i in range(len(cot_tokens) - self.max_children + 1))
        else:
            cot_seqs = set(tuple(cot_tokens[i:i+self.max_children]) for i in range(len(cot_tokens) - self.max_children + 1))
        # check if any of the sequences in cot_seqs is in the exclusion_set
        if not self.exclusion_set or len(self.exclusion_set) == 0:
            return False
        overlap = cot_seqs.intersection(self.exclusion_set)
        if overlap:
            # if len(overlap) > 0:
                # print(f"Found overlap with exclusion set: {overlap}")
            return True
        # for seq in self.exclusion_set:
        #     if seq in cot:
        #         return True
        return False
    
    def to_polish_notation(self, cot: str, funcs=None):
        """
        Convert the CoT string to Polish notation.
        """
        # remove parentheses and replace them with commas
        for f in funcs:
            cot = cot.replace(f"{f.__name__}{SEP}(", f"({SEP}{f.__name__}")
        return cot
    
    def create_dataset(self, num_samples=100):
        dataset = []
        for i in range(num_samples):
            # root = self.get_datapoint(name=f'root_{i}', verbose=False)
            # cot = root.get_CoT(max_iters=100, verbose=False)
            # output = str(root.compute(verbose=False))
            cot, output = self.get_datapoint(name=f'root_{i}', verbose=False)
            dataset.append((cot, output))
            if i % 10 == 0 and i > 0:
                print(f"Created {i} samples\t\t", end='\r')
        return dataset

    def make_test_sample(self, cot, output):
        """
        Create a test sample in the format similar to data_old.
        Args:
            cot (str): The CoT string.
            output (str): The output of the computation.
        Returns:
            dict: A dictionary containing the character, encoded, and result.
        """
        # the part we want to encode is only up to the first '='
        eq = cot.find('=')
        if eq != -1:
            equation = cot[:eq+1]
        else:
            raise ValueError("CoT string does not contain '=' character.")
        return {
            'character': cot,
            'encoded': self.tokenizer.encode(equation),
            # 'result': [self.vocab.index(str(output))]# if str(output) in self.vocab else 0]
            'result': self.tokenizer.encode(str(output)) #if str(output) in self.vocab else [0]
        }   

    def prepare_train_test_data(self, num_train=500,
                                num_test=100, 
                                use_exclusion_set=True,
                                # num_exclusion_samples=100
                                exclusion_rate=0.05,
                                ):
        """
        Prepare the training and test data.
        Args:
            num_train (int): Number of training samples to create.
            num_test (int): Number of test samples to create.
            use_exclusion_set (bool): Whether to use the exclusion set to filter out samples.
            num_exclusion_samples (int): Number of samples to create for the exclusion set if use_exclusion_set is True.
        Returns:
            None: The method modifies the train_data and test_data attributes of the dataset.
        """
        if use_exclusion_set and self.exclusion_set is None:
            # self.exclusion_set = self.make_exclusion_set(num_samples=num_exclusion_samples)
            self.exclusion_set = self.make_exclusion_set(sampling_rate=exclusion_rate)
        else:
            self.exclusion_set = set()

        num_samples = (num_train + num_test)*100  # we create twice as many samples to ensure we have enough for both train and test
        # self.data = self.create_dataset(num_samples=num_samples)
        # Filter out samples that contain any of the exclusion sequences (train data should not contain these)
        count_train = 0
        self.test_data = []
        self.train_data = []
        cot_encoded_lists = []

        for i in range(num_samples):
            t0 = time.time()
            if i % 100 == 0 and i > 0:
                print(f"Sample {i}: dt={time.time()-t0:.2g}; Prepared {count_train} training samples and {len(self.test_data)} test samples\t\t", end='\r')
            if count_train >= num_train and len(self.test_data) >= num_test:
                break
            cot, output = self.get_datapoint(name=f'root_{i}', verbose=False)
            if not self.in_exclusion_set(cot):
                if count_train < num_train:
                    count_train += 1
                    # self.train_data += self.tokenizer.encode(cot)
                    cot_encoded_lists.append(self.tokenizer.encode(cot))
            else:
                if len(self.test_data) < num_test:
                    self.test_data.append(self.make_test_sample(cot, output))     

        # Concatenate just once at the end
        self.train_data = [token for encoded_list in cot_encoded_lists for token in encoded_list]
        print(f"\nFinished preparing data. Total samples: {num_samples}.")
        print(f"Prepared {count_train} training samples and {len(self.test_data)} test samples.")
        self.num_train = count_train # the actual number of training samples created
        self.num_test = len(self.test_data)
        self.prepare_test_single_ops(num_samples=num_samples, num_test=num_test, use_exclusion_set=use_exclusion_set)
        
    def prepare_test_single_ops(self, num_samples=1000, num_test=100, use_exclusion_set=True):
        # single operation test data
        print(f"Preparing test samples for single operations...")
        self.test_single_ops = {}
        cnt = 0
        for f in self.funcs_to_use:
            self.test_single_ops[f.__name__] = []
            for i in range(num_samples):
                t0 = time.time()
                if i % 100 == 0 and i > 0:
                    print(f"Op: {f.__name__}, Sample {i}: dt={time.time()-t0:.2g}; Prepared {cnt} test samples.\t\t", end='\r')
                cnt = len(self.test_single_ops[f.__name__])
                if cnt >= num_test:
                    break
                cot, output = self.get_datapoint(name=f'{f.__name__}_test_{i}', funcs=[f])
                if use_exclusion_set and self.in_exclusion_set(cot):
                    # continue
                    self.test_single_ops[f.__name__].append(self.make_test_sample(cot, output))
            print(f"\nFinished preparing {len(self.test_single_ops[f.__name__])} test samples for operation {f.__name__}.")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        if not self.data:
            return f"ListOpsDataset(num_samples=0, no data loaded)"
        return f"ListOpsDataset(num_samples={len(self)}, max_depth={self.data[0][0].count(',') if self.data else 0})"

    def to_dict(self):
        """
        Convert the dataset to a dictionary format.
        """
        return {
            'train': self.train_data,
            'test': self.test_data,
            'test_single_ops': self.test_single_ops,
            'metadata': {
                'vocab': self.vocab,
                'max_depth': self.max_depth,
                'min_children': self.min_children,
                'max_children': self.max_children,
                'func_node_prob': self.func_node_prob,
                'all_funcs': [f.__name__ for f in self.all_funcs],
                'funcs_to_use': [f.__name__ for f in self.funcs_to_use],
                'input_set': self.input_set,
                'num_train': self.num_train,
                'num_test': self.num_test,
                'exclusion_set': list(self.exclusion_set) if self.exclusion_set else [],
                'exclude_permutation': self.exclude_permutation,
                'polish_notation': self.polish_notation,
            }
        }

    def make_name(self):
        """
        Generate a name for the dataset based on its parameters.
        """
        func_names = ','.join(sorted([f.__name__ for f in self.funcs_to_use]))
        return  f"ListOpsDataset{'_Polish' if self.polish_notation else ''}"+\
                f"_funcs({func_names})_depth{self.max_depth}"+\
                f"_args({self.min_children},{self.max_children})"+\
                f"_funcprob{self.func_node_prob:.2f}"+\
                f"_ntrain{self.num_train}"+\
                f"_ntest{self.num_test}"+\
                f"{'_excludeperm' if self.exclude_permutation else ''}"

        # f"_ntrain{len(self.train_data)}"+\

    def save(self, save_dir='../data/', filename=None):
        """
        Save the dataset to a file.
        """
        import pickle
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = self.make_name() + '.pkl'
        path = os.path.join(save_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        print(f"Dataset saved to {path}")

    # def save_hdf5(self, save_dir='../data/', filename=None):
    #     """
    #     Save the dataset to a file in hdf5 format.
    #     """
    #     import h5py
    #     import os
    #     import pickle
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir, exist_ok=True)
    #     if filename is None:
    #         filename = self.make_name() + '.h5'
    #     path = os.path.join(save_dir, filename)
    #     d = self.to_dict()
    #     with h5py.File(path, 'w') as f:
    #         # Create datasets for train and test data
    #         f.create_dataset('train', data=d['train'])
    #         # f.create_dataset('test', data=d['test'])
    #         # all the data is small, so use attrs and pickle to save it
    #         # f.attrs['test'] = pickle.dumps(d['test'])

    #         # for k, v in d.items():
    #         #     f.attrs[k] = pickle.dumps(v)
    #         # save the whole metadata as a dictionary
    #         # Store the entire metadata dict as a pickled bytes object in a single attribute
    #         f.attrs['metadata'] = pickle.dumps(d['metadata'])

    #     print(f"Dataset saved to {path}")

    # def load(self, path):
    #     """
    #     Load the dataset from a file.
    #     """
    #     # import pickle
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)
    #     self.train_data = data['train']
    #     self.test_data = data['test']
    #     self.vocab = data['metadata']['vocab']
    #     self.max_depth = data['metadata']['max_depth']
    #     self.min_children = data['metadata']['min_children']
    #     self.max_children = data['metadata']['max_children']
    #     self.func_node_prob = data['metadata']['func_node_prob']
    #     # self.all_funcs = [getattr(fs, f) for f in data['metadata']['all_funcs']]
    #     # self.funcs_to_use = [getattr(fs, f) for f in data['metadata']['funcs_to_use']]
    #     self.all_funcs = [f for f in data['metadata']['all_funcs']]
    #     self.funcs_to_use = [f for f in data['metadata']['funcs_to_use']]
    #     self.input_set = data['metadata'].get('input_set', list(range(10)))
    #     self.exclusion_set = data['metadata'].get('exclusion_set', [])
    #     self.num_train = len(self.train_data)
    #     self.num_test = len(self.test_data)

    #     # Reinitialize the tokenizer
    #     self.tokenizer = Tokenizer(self.vocab, sep=',')

    #     print(f"Dataset loaded from {path}")
    #     return self
    # def load_hdf5(self, path):
    #     """
    #     Load the dataset from a file in hdf5 format.
    #     """
    #     # import h5py
    #     # import pickle
    #     with h5py.File(path, 'r') as f:
    #         self.train_data = f['train'][:]
    #         self.test_data = f['test'][:]
    #         # Load the metadata from the attribute
    #         metadata = pickle.loads(f.attrs['metadata'])
    #         self.vocab = metadata['vocab']
    #         self.max_depth = metadata['max_depth']
    #         self.min_children = metadata['min_children']
    #         self.max_children = metadata['max_children']
    #         self.func_node_prob = metadata['func_node_prob']
    #         # self.all_funcs = [getattr(fs, f) for f in metadata['all_funcs']]
    #         # self.funcs_to_use = [getattr(fs, f) for f in metadata['funcs_to_use']]
    #         self.all_funcs = [f for f in metadata['all_funcs']]
    #         self.funcs_to_use = [f for f in metadata['funcs_to_use']]
    #         self.exclusion_set = metadata.get('exclusion_set', [])
    #         self.num_train = len(self.train_data)
    #         self.num_test = len(self.test_data)

    #         # Reinitialize the tokenizer
    #         self.tokenizer = Tokenizer(self.vocab, sep=',')

    #     print(f"Dataset loaded from {path}")
    #     return self
    # def __str__(self):
    #     return f"""ListOpsDataset(num_train={self.num_train}, 
    #         num_test={self.num_test}, 
    #         max_depth={self.max_depth}, 
    #         min_children={self.min_children}, 
    #         max_children={self.max_children}, 
    #         func_node_prob={self.func_node_prob}, 
    #         funcs_to_use={[f.__name__ for f in self.funcs_to_use]})"""
