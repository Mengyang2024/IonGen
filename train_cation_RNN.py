from RNN import model

if __name__ == '__main__':
    model.train(path_csv='data/unique_cations.csv', path_save_model='data/rnn_model_cation', num_epochs=100)
    
