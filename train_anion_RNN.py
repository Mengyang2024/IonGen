from RNN import model

if __name__ == '__main__':
    model.train(path_csv='data/unique_anions.csv', path_save_model='data/rnn_model_anion', num_epochs=100)
    
