import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# my function definition
def my_log_sum_exp(_x):
    """
    :param _x:
    :return:
    """
    return np.log(np.sum(np.exp(_x), axis=-1, keepdims=True))


# generate data and dataset
def generate_dataset(_dim, _len, _percent=0.8):
    """
    :param _dim: size of each element: (_dim,)
    :param _len: size of the length of dataset
    :param _percent:
    :return: train and var dataset
    """
    _dim = max(_dim, 1)  # dim > 1
    _len = max(_len, 100)  # len > 100
    train_len = max(int(_len * _percent), 1)
    var_len = max(int(_len - train_len), 1)
    return np.random.randn(train_len, _dim), np.random.randn(var_len, _dim)


class MyNetByTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        super(MyNetByTorch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float64)

    def forward(self, x):
        """
        :param x: input of the net
        :return:
        """
        x = torch.tensor(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_by_torch(_train, _net, _epochs=15):
    """
    :param _train: dataset
    :param _net: network
    :param _epochs:
    :return: no return
    """
    optimizer = optim.Adam(_net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    target = torch.tensor(my_log_sum_exp(_train))

    for _epoch in range(_epochs):
        _output = _net(_train)
        _loss = criterion(_output, target)

        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        if (_epoch + 1) % 10 == 0:
            print(f'Epoch [{_epoch + 1}/{_epochs}], Loss: {_loss.item():.4f}')


def evaluate(_net, _var, epsilon=1e-3):
    """
    :param _net:
    :param _var: var dataset
    :param epsilon: threshold
    :return: accuracy of the model
    """
    return np.mean(np.abs((_net(_var).detach().numpy() - my_log_sum_exp(_var)) < epsilon).astype(np.float64))


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    LENGTH = 100000  # size of the dataset

    # net parameters
    INPUT_SIZE = 10
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 1

    THRESHOLD = 1e-3  # verify whether the predict value is close to the ground truth
    
    train, var = generate_dataset(INPUT_SIZE, LENGTH)
    my_result = my_log_sum_exp(train)
    torch_result = torch.logsumexp(torch.tensor(train), dim=-1, keepdim=True).numpy()
    print("result by my log sum exp compared", np.abs(np.mean(my_result - torch_result)) < THRESHOLD)
    net = MyNetByTorch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    train_by_torch(train, net)
    result = evaluate(net, var)
    print("torch network accuracy: ", result)
