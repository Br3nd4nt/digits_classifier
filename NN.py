from sklearn.neural_network import MLPClassifier
from pandas import read_csv
from numpy import random
class NeuralNetwork:
    def __init__(self, input_size = 784, output_size = 10):
        self.input_size = input_size
        self.output_size = output_size
        self.nn = MLPClassifier(hidden_layer_sizes=(self.input_size // 2, self.input_size // 4,), activation='tanh', verbose=True,tol=1e-6)
    

    # https://www.kaggle.com/datasets/oddrationale/mnist-in-csvg
    def train(self, dataset_path = './mnist_train.csv'):
        df = read_csv(dataset_path, delimiter=',').to_numpy()
        print(df.shape)
        y_train = df[:, 0]
        X_train = df[:, 1:]
        self.nn.fit(X_train, y_train)
    
    def score(self, test_dataset_path = './mnist_test.csv'):
        df = read_csv(test_dataset_path, delimiter=',').to_numpy()
        y_test = self.apply_noise(df[:, 0])
        X_test = df[:, 1:]
        return self.nn.score(X_test, y_test)
    
    def apply_noise(self, matrix, power = 10): 
        return matrix + random.random(matrix.shape) * power

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.train()
    from joblib import dump
    dump(nn, "v2.joblib")
