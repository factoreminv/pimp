import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

#import helper

# moves your model to train on your gpu if available else it uses your cpu
device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])



# Download and load the training data
train_set = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # Convolutional Neural Network Layer
        self.convolutional_neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1),  # (N, 1, 28, 28)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=6 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.convolutional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    model = Network()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 10  # The total number of iterations

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # prep model for training
        model.train()
        train_loss = 0

        for idx, (images, labels) in enumerate(trainloader):

            # Send these >>> To GPU
            images = images.to(device)
            labels = labels.to(device)

            # Training pass
            optimizer.zero_grad()

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            # prep model for evaluation
            model.eval()
            test_loss = 0
            accuracy = 0

            # Turn off the gradients when performing validation.
            # If we don't turn it off, we will comprise our networks weight entirely
            with torch.no_grad():
                for images, labels in testloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    log_probabilities = model(images)
                    test_loss += criterion(log_probabilities, labels)

                    probabilities = torch.exp(log_probabilities)
                    top_prob, top_class = probabilities.topk(1, dim=1)
                    predictions = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(predictions.type(torch.FloatTensor))

            train_losses.append(train_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))

            print("Epoch: {}/{}  ".format(epoch + 1, epochs),
                  "Training loss: {:.4f}  ".format(train_loss / len(trainloader)),
                  "Testing loss: {:.4f}  ".format(test_loss / len(testloader)),
                  "Test accuracy: {:.4f}  ".format(accuracy / len(testloader)))

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.grid()
