from MCTS import tree_anion

if __name__ == '__main__':
    MCtree = tree_anion.BuildTree(rnn_model_path='data/rnn_model_anion')
    MCtree.search(stock_path='data/pubchem_anions.csv', saved_anions_path='generated_anions.csv', num_loops=100)
    
