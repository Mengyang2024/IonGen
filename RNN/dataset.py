"""Load and process dataset for RNN training.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from RNN.utils import *


def from_csv(path: str) -> 'Dataset':
    """
    Read dataset from a .csv file. The .csv file needs to contain a column named "smiles" 
    and only the column of "smiles" will be loaded.

    :param path: the path to the .csv file.
    :return: a numpy array of the loaded SMILES strings list
    """
    dataset = Dataset(pd.read_csv(path)['smiles'].to_numpy())
    return dataset

class Dataset:
    """
    Load a training dataset and process the loaded dataset. 

    :param dataset: a numpy array contains a list of SMILES strings
    """

    def __init__(self, dataset: np.ndarray) -> None:
        self.dataset: np.ndarray = dataset

    def get_total_symbol_list(self) -> list:
        """
        Split all the SMILES strings in the loaded data set into individual symbols.
        Reconstruct a total symbol list by removing duplicates, 
        which will be used as a dictionary for translating symbol characters to integers.

        :return: a total symbol list
        """
        symbol_list = ['$', '^']
        for smi in self.dataset:
            symbols = split_SMILES(smi)
            for symbol in symbols:
                if symbol not in symbol_list:
                    symbol_list.append(symbol)
        return symbol_list
    
    def split_dataset(self) -> [list, list]:
        """
        Split all the SMILES strings in the loaded dataset into symbol-constructed lists.
        Reconstruct a total symbol list by removing duplicates

        :return: a list contains all the symbol-constructed lists
        :return: a total symbol list
        """
        splitted_smis = []
        total_symbol_list = ['$', '^']
        for smi in self.dataset:
            symbols = split_SMILES(smi)
            splitted_smis.append(symbols)
            for symbol in symbols:
                if symbol not in total_symbol_list:
                    total_symbol_list.append(symbol)
        return splitted_smis, total_symbol_list
        
    def process(self, max_len: int = 110):
        """
        To translate all the SMILES strings in the loaded dataset into integer numpy arrays.

        :param max_len: the length of the integer numpy arrays to be translated. default = 110

        :return: two integer numpy arrays. The first one, which the translated integer numpy array, 
        will used as features for RNN training and the second array, which is offset by one 
        element compared to the first one, will be used as target for RNN training.
        """
        X = []
        Y = []
        splitted_dataset, total_symbol_list = self.split_dataset()
        for splitted_smi in splitted_dataset:
            x = []
            for symbol in splitted_smi:
                x.append(total_symbol_list.index(symbol))
            y = x[1:]
            X.append(np.pad(x, (0, max_len - len(x)), 'constant', constant_values=(0, 0)))
            Y.append(np.pad(y, (0, max_len - len(y)), 'constant', constant_values=(0, 0)))
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

class DataPrep(tf.keras.utils.Sequence):
    """
    To read a dataset from .csv file and prepare for the training.

    :param path: path to the .csv file
    :batch size: the size of each batch for training 
    """
    def __init__(self, path: str, batch_size: int = 256) -> None:
        self.dataset = from_csv(path)
        self.x, self.y = self.dataset.process()
        self.batch_size = batch_size
        self.total_symbol_list = self.dataset.get_total_symbol_list()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = np.array([to_one_hot(x, len(self.total_symbol_list)) for x in batch_x])
        batch_Y = np.array([to_one_hot(y, len(self.total_symbol_list)) for y in batch_y])
        return batch_X, batch_Y

