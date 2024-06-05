"""The monte carlo tree
"""
from RNN import model
import numpy as np
import pandas as pd
from rdkit import Chem
import math
import graphviz
from multiprocessing import get_context


class Node:
    """
    To define a node in the monte carlo tree.

    :param current_SMILES: the SMILES symbol list of the current node
    :param parent: the parent of the current node
    """
    def __init__(self, current_smiles: list = None, parent: "Node" = None) -> None:
        self.current_smiles = current_smiles
        self.parent = parent
        self.visits = 0
        self.total_reward = 0
        self.children = []

    def ucb1(self) -> float:
        """
        To calculate the UCB score of the current node

        :return: UCB1 score
        """
        if self.parent is None:
            ucb1 = 0
        elif self.visits == 0:
            ucb1 = 0
        else:
            ucb1 = self.total_reward / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return ucb1

    def show(self) -> graphviz.Digraph():
        """
        To show the details and position of the current node in picture.

        :return: a graphviz.Digraph() object to show the picture
        """
        node_current = self
        dot = graphviz.Digraph()
        while True:
            dot.node('select node\nsmiles=%s\nreward=%f\nvisits=%d\nUCB1=%f' 
                     % (str(node_current.current_smiles),
                            node_current.total_reward,
                            node_current.visits,
                            node_current.ucb1()))
            if node_current.parent is not None:
                for child in node_current.parent.children:
                    if child == node_current:
                        dot.edge('select node\nsmiles=%s\nreward=%f\nvisits=%d\nUCB1=%f' % (
                            str(node_current.parent.current_smiles),
                            node_current.parent.total_reward,
                            node_current.parent.visits,
                            node_current.parent.ucb1()),
                                 'select node\nsmiles=%s\nreward=%f\nvisits=%d\nUCB1=%f' 
                                 % (str(child.current_smiles),
                                    child.total_reward,
                                    child.visits,
                                    child.ucb1()), )
                    else:
                        dot.edge('select node\nsmiles=%s\nreward=%f\nvisits=%d\nUCB1=%f' % (
                            str(node_current.parent.current_smiles),
                            node_current.parent.total_reward,
                            node_current.parent.visits,
                            node_current.parent.ucb1()),
                                 'smiles=%s\nreward=%f\nvisits=%d\nUCB1=%f' 
                                 % (str(child.current_smiles),
                                    child.total_reward,
                                    child.visits,
                                    child.ucb1()), )
                node_current = node_current.parent
            else:
                break
        return dot
    
class BuildTree:
    def __init__(self, rnn_model_path: str) -> None:
        self.model = model.LoadFromFile(rnn_model_path)
    
    def selection(self, input_node: "Node") -> "Node":
        """
        Traverse down the tree based on UCB1 until reaching a leaf node

        :param input_node: the input node
        
        :return: return the selected node
        """
        while True:
            if len(input_node.children) == 0:
                break
            max_ucb1_index = np.argmax([child.ucb1() for child in input_node.children])
            input_node = input_node.children[max_ucb1_index]
        return input_node
    
    def expansion(self, input_node: "Node") -> "Node":
        """
        Create a new child node from the selected leaf node

        :param input_node: the input node to expansion

        :return: a node with expanded childern nodes
        """

        expansion_policy = self.model.predict_next_possible_symbols
        possible_next_symbols = expansion_policy(input_node.current_smiles) 
        for next_symbol in possible_next_symbols:
            next_smiles = input_node.current_smiles + [next_symbol]
            new_node = Node(current_smiles=next_smiles, parent=input_node)
            input_node.children.append(new_node)
        return input_node
    
    def simulation(self, input_node:"Node", stock: list, path_to_store_cations: "open") -> dict:
        """
        Simulate a random rollout from the selected leaf node

        :param input_node: the input node to run rollout
        :stock: a stock of existing cations, which will be used to check the novelty of the SMILES
        :file_to_store_cations: a file to save generated cations

        :return a dict for nodes with scores
        """

        rollout_policy = self.model.predict_complete_SMILES
        node_SMILES_dict = {}
        for child in input_node.children:
            node_SMILES_dict[child] = []
            with get_context("spawn").Pool(10) as pool1:
                smi = child.current_smiles.copy()
                node_SMILES_dict[child] = pool1.map(rollout_policy, [smi for _ in range(10)])
            print('generate cations %s'%node_SMILES_dict[child])
        score_dict, cations = reward_score(node_SMILES_dict, stock)
        with open(path_to_store_cations, 'a') as file1:
            for cation in cations:
                print(cation, file=file1)
        return score_dict
    
    def backpropagation(self, score_dict: dict) -> "Node":
        """
        Update the scores of the nodes along the path

        :param score_dict: a dict to store reward scores for each node
        """

        for edge_node in score_dict:
            score = score_dict[edge_node]
            current_node = edge_node
            while True:
                current_node.visits += 1
                current_node.total_reward += score
                current_node = current_node.parent
                node = current_node
                if current_node.parent is None:
                    current_node.visits += 1
                    current_node.total_reward += score
                    break
        return node
    
    def search(self, stock_path: str, 
               saved_cations_path: str, 
               num_loops: int,
               root_SMILES: list = ['^']) -> None:
        """
        To run the tree search.

        :param stock_path: a path to the stock of existing cations, which will be used to check the novelty of the SMILES
        :param saved_cations_path: a path to save generated cations
        :param num_loops: number of loops to run
        :root_SMILES: the SMILES of the root node 
        """

        stock = list(pd.read_csv(stock_path)['smiles'])
        node = Node(current_smiles=root_SMILES)
        for _ in range(num_loops):
            print('loop_%d'%_)
            node = self.selection(node)
            print('current path = %s'%node.current_smiles)
            node = self.expansion(node)
            score_dict = self.simulation(node, stock, saved_cations_path)
            node = self.backpropagation(score_dict)
        
def reward_score(node_SMILES_dict: dict, stock: list) -> (dict, list):
    """
    To calculate the reward scores of a series of node
    according to their generated complete SMILES 

    :param node_SMILES_dict: a series of node with their generated complete SMILES 
    :stock: a stock of existing cations, which will be used to check the novelty of the SMILES
    
    :return: a dict to store the calculated reward scores for each input node
    :return: a list of found valid SMILES
    """

    score_dict = {}
    cations = []
    for node in node_SMILES_dict:
        score_dict[node] = 0
        smiles_checking_result = [check_smiles(smi, stock) for smi in node_SMILES_dict[node]]
        for item in smiles_checking_result:
            if item != 'NaN':
                cations.append(item)
                score_dict[node] = 1
    return score_dict, list(set(cations))

def check_smiles(input_SMILES: str, stock: list) -> str:
    """
    To check if a SMILES available.

    :param input_SMILES: the SMILES to be checked
    :stock: a stock of existing cations, which will be used to check the novelty of the SMILES
    
    :return: if available, return the input SMILES, else retrun "NaN"
    """
    try:
        mol = Chem.MolFromSmiles(input_SMILES)
    except Exception:
        return 'NaN'
    if mol != None:
        for at in mol.GetAtoms():
            if at.GetNumRadicalElectrons() != 0:
                return 'NaN'
            else:
                continue
        canonical_smiles = Chem.MolToSmiles(mol)
        if canonical_smiles in stock:
            return 'NaN'
        else:
            pass
        if canonical_smiles.count('+]') == 1 and canonical_smiles.count('-]') == 0 \
           and canonical_smiles.count('+2]') == 0 and canonical_smiles.count('+3]') == 0 \
           and canonical_smiles.count('-2]') == 0 and canonical_smiles.count('-3]') == 0:
            print('find cations %s'%input_SMILES)
            return canonical_smiles
        else:
            return 'NaN'
    else:
        return 'NaN'
