"""RNN model
"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
from RNN import dataset
from RNN.utils import *
import os
import time
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train(path_csv: str, path_save_model: str, num_epochs: int) -> None:
    """
    Train the RNN model using dataset from a .csv file. The .csv file needs to contain a column named "smiles" 
    and only the column of "smiles" will be used.

    :param path_csv: path to the .csv file
    :param path_save_model: a path to save the trained model
    :param num_epochs: number of epochs
    """

    training_data = dataset.DataPrep(path_csv)
    total_symbol_list = training_data.total_symbol_list
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(256, input_shape=(None, len(total_symbol_list)), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GRU(256, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(len(total_symbol_list), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(training_data, epochs=num_epochs)
    model.save(path_save_model)
    if path_save_model[-1] == '/':
        path_to_tottal_symbol_list = '%stotal_symbol_list'%path_save_model
    else:
        path_to_tottal_symbol_list = '%s/total_symbol_list'%path_save_model
    with open(path_to_tottal_symbol_list, 'w') as f:
        for symbol in total_symbol_list:
            print(symbol, file=f, end=' ')

class LoadFromFile:
    """
    Use a pretrained RNN model to predict complete SMILES strings 
    or guess the next symbol from incomplete SMILES symbol list.

    :param path: path to the pretrained RNN model
    """

    def __init__(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
        if path[-1] == '/':
            self.path_to_total_symbol_list = '%stotal_symbol_list'%path
        else:
            self.path_to_total_symbol_list = '%s/total_symbol_list'%path

        with open(self.path_to_total_symbol_list, 'r') as f:
            self.total_symbol_list = f.readlines()[0].split()
    
    def predict_next_possible_symbols(self, symbol_list: list) -> list:
        """
        Guess a list of next possible symbols
        
        :param symbol_list: the incomplete SMILES string symbol list
        :return: next possible symbols
        """
        input = Translater(symbol_list, self.path_to_total_symbol_list).to_one_hot()
        input = np.array([input])
        y = np.asarray(self.model.predict(input, verbose=None)[0][len(symbol_list)-1]).astype('float64')
        y = y / np.sum(y)
        multinomial_distribution = np.random.multinomial(1, y, 50)
        possible_selection = list(set([np.where(item == 1)[0][0] for item in multinomial_distribution]))
        possible_next_symbols = [self.total_symbol_list[symbol_id] for symbol_id in possible_selection]
        return possible_next_symbols
    
    def predict_next_possible_symbol(self, symbol_list: list) -> str:
        """
        Guess one next possible symbol
        
        :param symbol_list: the incomplete SMILES string symbol list
        :return: next possible symbol
        """

        input = Translater(symbol_list, self.path_to_total_symbol_list).to_one_hot()
        input = np.array([input])
        y = np.asarray(self.model.predict(input, verbose=None)[0][len(symbol_list)-1]).astype('float64')
        y = y / np.sum(y)
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        multinomial_distribution = np.random.multinomial(1, y, 1)
        next_symbol = self.total_symbol_list[np.argmax(multinomial_distribution)]
        return next_symbol
    
    def predict_complete_SMILES(self, symbol_list: list, max_atom_num: int = 80) -> str:
        """
        Predict a complete SMILES string

        :param symbol_list: the incomplete SMILES string symbol list
        :param max_atom_num: the max atom number of the generated molecule
        :return: a complete SMILES string
        """
        possible_SMILES_list = symbol_list
        while True:
            next_symbol = self.predict_next_possible_symbol(possible_SMILES_list)
            if next_symbol == '$':
                break
            if len(possible_SMILES_list) > max_atom_num:
                break
            possible_SMILES_list += [next_symbol]
        possible_SMILES = ''.join(possible_SMILES_list[1:])
        return possible_SMILES
    
    
