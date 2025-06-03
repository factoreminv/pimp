import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cvxpy as cp

import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms
from time import time
import matplotlib.pyplot as plt

from PIL import Image

def closest_permutation(X):
    cost = -X
    row_ind, col_ind = linear_sum_assignment(cost)
    n, m = X.shape
    P = np.zeros((n, m))
    P[row_ind, col_ind] = 1
    return P


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.log_softmax(self.fc3(h2), dim=1)
        return h3

    def change_wm2(self, ideal):

        # with torch.no_grad():
        #     self.fc2.weight.copy_(torch.from_numpy(ideal[np.random.permutation(ideal.shape[0]),:]))

        # plt.imshow(self.fc2.weight.detach().numpy())
        # plt.show()

        n = self.fc1.weight.shape[0]
        P1 = cp.Variable((n, n))
        P2 = cp.Variable((n, n))


        objective = cp.Minimize(cp.norm(ideal - P1 @ self.fc2.weight.detach().numpy(), 2))
        constraints = [cp.sum(P1, axis=0) == np.ones(n),
                       cp.sum(P1, axis=1) == np.ones(n),
                       P1 >= 0]

        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver='SCS')
        P1 = closest_permutation(P1.value)

        print("found P1")

        # plt.imshow(P1 @ self.fc2.weight.detach().numpy())
        # plt.show()

        objective = cp.Minimize(cp.norm(ideal - P2 @ self.fc2.weight.detach().numpy() @ np.linalg.inv(P1), 2))
        constraints = [cp.sum(P2, axis=0) == np.ones(n),
                       cp.sum(P2, axis=1) == np.ones(n),
                       P2 >= 0]

        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver='SCS')
        P2 = closest_permutation(P2.value)

        print("found P2")

        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(P1 @ self.fc1.weight.detach().numpy()))
            self.fc1.bias.copy_(torch.from_numpy(P1 @ self.fc1.bias.detach().numpy()))

            self.fc2.weight.copy_(torch.from_numpy(P2 @ self.fc2.weight.detach().numpy() @ np.linalg.inv(P1)))
            self.fc2.bias.copy_(torch.from_numpy(P2 @ self.fc2.bias.detach().numpy()))

            self.fc3.weight.copy_(torch.from_numpy(self.fc3.weight.detach().numpy() @ np.linalg.inv(P2)))

        print("updated weights and biases")



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=True,
    transform=transform
)
tvset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=False,
    transform=transform
)

valset, testset = torch.utils.data.random_split(tvset, [3000, 7000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

def train(lr, maxepochs):
    model = MyNet()
    params = list(model.parameters())

    optimizer = optim.SGD(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    criterion = nn.NLLLoss()

    time0 = time()
    model.train()

    losses = []

    e= 0

    while e < maxepochs:
        e += 1
        running_loss = 0.0

        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            out = model(images)

            loss_ce = criterion(out, labels)

            loss = loss_ce

            loss.backward()
            optimizer.step()

            running_loss += loss_ce.item()

        print(f"{lr}   Epoch {e + 1}/{500} - Classification Loss: {running_loss / len(trainloader):.4f}")


        losses.append(running_loss / len(trainloader))

        val = validate(model=model, dataset=valset)
        if val > 0.99:
            break

        scheduler.step()

    print(f"\nTraining Time (minutes): {(time() - time0) / 60:.2f}")

    plt.semilogy(losses)
    plt.title("classification loss vs epochs")
    plt.show()

    # np.savetxt("weights.txt", model.get_weights()[1].detach().numpy())

    return validate(model=model, dataset=testset), model

def validate(model, dataset):
    model.eval()

    correct_count, all_count = 0, 0

    valloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    for images, labels in valloader:
        images = images.view(images.shape[0], -1)

        with torch.no_grad():
            out = model(images)  # returns log-softmax

        ps = torch.exp(out)  # shape: (batch_size, 10)

        ps_cpu = ps.cpu().numpy()
        labels_cpu = labels.cpu().numpy()

        pred_labels = np.argmax(ps_cpu, axis=1)

        correct_count += (pred_labels == labels_cpu).sum()
        all_count += len(labels_cpu)

    # print(f"\nNumber of Images Tested = {all_count}")
    # print(f"Model Accuracy = {correct_count / all_count:.4f}")

    return correct_count / all_count

if __name__ == '__main__':
    lena = "/Users/ozgursoysal/PycharmProjects/ARIKAN RG/UVsparse/lena.jpg"
    img = Image.open("lena.jpg").convert('L')

    img = np.array(img)[::16, ::16]

    plt.matshow(img)
    plt.show()


    score, model = train(lr=0.05, maxepochs = 5)

    print("pre-jump score: ", score)

    plt.imshow(model.fc2.weight.detach().numpy())
    plt.title("prejump fc2 weights")
    plt.show()

    model.change_wm2(img)

    # wm = model.fc2.weight.data
    # plt.matshow(wm)
    # plt.show()

    print("post-jump score", validate(model=model, dataset=testset))

    plt.imshow(model.fc2.weight.detach().numpy())
    plt.title("postjump fc2 weights")
    plt.show()