from MCTS import tree_cation

if __name__ == '__main__':
    MCtree = tree_cation.BuildTree(rnn_model_path='data/rnn_model_cation')
    MCtree.search(stock_path='data/pubchem_cations.csv', saved_cations_path='generated_cations.csv', num_loops=100)
    
