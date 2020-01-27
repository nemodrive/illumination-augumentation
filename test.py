import torch
import torch.nn as nn

if __name__ == '__main__':
    loss = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    target = sigmoid(torch.zeros(10))
    prediction = sigmoid(torch.ones(10))
    score = loss(prediction, prediction)
    print(score)