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

from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset



def randomperm(n):
    I = np.eye(n)
    return I[np.random.permutation(n), :]


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 3)

        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.log_softmax(self.fc4(h3), dim=1)
        return h4

    def randomjump(self):

        P1 = randomperm(32)
        P2 = randomperm(32)

        plt.imshow(P1)
        plt.show()

        D1 = np.diag(np.abs(np.random.randn(32)/2) + 1)
        D1 = D1 / np.linalg.det(D1) ** (1 /  32)
        # print(D1)
        # print(np.linalg.det(D1))
        D2 = np.diag(np.abs(np.random.randn(32)/2) + 1)
        D2 = D2 / np.linalg.det(D2) ** (1 / 32)
        # print(np.linalg.det(D2))

        M1 = D1 @ P1
        M2 = D2 @ P2

        plt.imshow(M1)
        plt.show()
        plt.imshow(M2)
        plt.show()

        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(M1 @ self.fc1.weight.detach().numpy()))
            self.fc1.bias.copy_(torch.from_numpy(M1 @ self.fc1.bias.detach().numpy()))

            self.fc2.weight.copy_(torch.from_numpy(M2 @ self.fc2.weight.detach().numpy() @ np.linalg.inv(M1)))
            self.fc2.bias.copy_(torch.from_numpy(M2 @ self.fc2.bias.detach().numpy()))

            self.fc3.weight.copy_(torch.from_numpy(self.fc3.weight.detach().numpy() @ np.linalg.inv(M2)))

        # print("updated weights and biases")



wine = load_wine()
X, y = wine.data, wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

trainset = TensorDataset(X_train_tensor, y_train_tensor)
testset = TensorDataset(X_test_tensor, y_test_tensor)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)


def train(lr, maxepochs, trainlength):

    np.random.seed(1)

    model = MyNet()
    params = list(model.parameters())

    optimizer = optim.SGD(params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    criterion = nn.NLLLoss()

    time0 = time()
    model.train()

    losses = []

    vallosses = []

    gradnorms = []

    e= 0

    stopepoch = 0

    while e < maxepochs:
        e += 1
        running_loss = 0.0

        total_norm = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            out = model(images)

            loss_ce = criterion(out, labels)

            loss = loss_ce

            for l in model.layers:
                loss += 0.005*l.weight.norm(p=1)

            optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

            for g in optimizer.param_groups:
                if total_norm != 0:
                    g['lr'] = lr/np.sqrt(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss_ce.item()
        print(f"{lr}   Epoch {e + 1}/{maxepochs} - Classification Loss: {running_loss / len(trainloader):.4f}")


        losses.append(running_loss / len(trainloader))

        val = validate(model=model, dataset=testset)
        # print(val*100)
        vallosses.append(val*100)

        total_norm = total_norm ** 0.5
        gradnorms.append(total_norm)

        if val > 0.9999:
            print("TOTAL DOMINATION")
            stopepoch = e
            break

        plt.imshow(model.fc2.weight.detach().numpy())
        plt.title("2")
        plt.colorbar()
        plt.show()

        plt.imshow(model.fc3.weight.detach().numpy())
        plt.title("3")
        plt.colorbar()
        plt.show()

        plt.imshow(model.fc4.weight.detach().numpy())
        plt.title("4")
        plt.colorbar()
        plt.show()

        if e > 0 and e % trainlength == 0:
            optimizer.zero_grad()
            # plt.imshow(model.fc2.weight.detach().numpy())
            # plt.show()
            val123 = validate(model=model, dataset=testset)
            print("prejump", val123)
            model.randomjump()

            for g in optimizer.param_groups:
                g['lr'] = lr
            # vallosses.append(val*100)
            print("postjump", validate(model=model, dataset=testset))
            optimizer.zero_grad()

    print(f"\nTraining Time (minutes): {(time() - time0) / 60:.2f}")

    plt.semilogy(losses)
    a = trainlength
    while a < stopepoch:
        plt.vlines(x=a, ymin=np.min(losses), ymax=np.max(losses), colors='r', linestyles='--')
        a += trainlength
    plt.title(f"training loss vs epochs - trainperiod:{trainlength}")
    plt.show()

    plt.plot(vallosses)
    a = trainlength
    while a < stopepoch:
        plt.vlines(x=a, ymin=np.min(vallosses), ymax=np.max(vallosses), colors='r', linestyles='--')
        a += trainlength
    plt.title("test accuracy vs epochs")
    plt.show()

    plt.plot(gradnorms)
    a = trainlength
    while a < stopepoch:
        plt.vlines(x=a, ymin=np.min(gradnorms), ymax=np.max(gradnorms), colors='r', linestyles='--')
        a += trainlength
    plt.title("total gradient norm vs epochs")
    plt.show()

    plt.semilogy(losses[:10*trainlength])
    a = trainlength
    while a < stopepoch:
        plt.vlines(x=a, ymin=np.min(losses), ymax=np.max(losses), colors='r', linestyles='--')
        a += trainlength
    plt.title(f"training loss vs epochs - trainperiod:{trainlength}")
    plt.show()


    # np.savetxt("weights.txt", model.get_weights()[1].detach().numpy())

    return validate(model=model, dataset=testset), model

def validate(model, dataset):
    model.eval()

    correct_count, all_count = 0, 0

    valloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

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
    torch.manual_seed(12032025)
    train(lr = 15, maxepochs = 20000, trainlength = 20001)