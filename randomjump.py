from sched import scheduler

import numpy.linalg
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cvxpy as cp

import torch.optim as optim
from torchvision import datasets, transforms
from time import time
import matplotlib.pyplot as plt

from PIL import Image

import copy


def randomperm(n):
    I = np.eye(n)
    return I[np.random.permutation(n), :]


class MyNet(nn.Module):
    def __init__(self, variance):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.var = variance

        self.record = []

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.log_softmax(self.fc4(h3), dim=1)
        return h4

    def randomjump(self):

        P1 = randomperm(256)
        P2 = randomperm(128)
        # P3 = randomperm(128)
        # P4 = randomperm(128)
        # P5 = randomperm(128)
        # P6 = randomperm(128)
        # P7 = randomperm(128)
        # P8 = randomperm(128)

        # plt.imshow(P1)
        # plt.show()

        D1 = np.diag(np.abs(np.random.randn(256) * self.var) + 0.2)
        D1 = D1 / np.linalg.det(D1) ** (1 / 256)
        # print(D1)
        # print(np.linalg.det(D1))
        D2 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        D2 = D2 / np.linalg.det(D2) ** (1 / 128)
        # print(np.linalg.det(D2))

        # D3 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        # D3 = D3 / np.linalg.det(D3) ** (1 / 128)
        #
        # D4 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        # D4 = D4 / np.linalg.det(D4) ** (1 / 128)
        #
        # D5 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        # D5 = D5 / np.linalg.det(D5) ** (1 / 128)
        #
        # D6 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        # D6 = D6 / np.linalg.det(D6) ** (1 / 128)
        #
        # D7 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        # D7 = D7 / np.linalg.det(D7) ** (1 / 128)
        #
        # D8 = np.diag(np.abs(np.random.randn(128) * self.var) + 0.2)
        # D8 = D8 / np.linalg.det(D8) ** (1 / 128)

        M1 = D1 @ P1
        M2 = D2 @ P2
        # M3 = D3 @ P3
        # M4 = D4 @ P4
        # M5 = D5 @ P5
        # M6 = D6 @ P6
        # M7 = D7 @ P7
        # M8 = D8 @ P8

        # plt.imshow(M1)
        # plt.show()
        # plt.imshow(M2)
        # plt.show()

        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(M1 @ self.fc1.weight.detach().numpy()))
            self.fc1.bias.copy_(torch.from_numpy(M1 @ self.fc1.bias.detach().numpy()))

            self.fc2.weight.copy_(torch.from_numpy(M2 @ self.fc2.weight.detach().numpy() @ np.linalg.inv(M1)))
            self.fc2.bias.copy_(torch.from_numpy(M2 @ self.fc2.bias.detach().numpy()))

            self.fc3.weight.copy_(torch.from_numpy(self.fc3.weight.detach().numpy() @ np.linalg.inv(M2)))
            # self.fc3.bias.copy_(torch.from_numpy(M3 @ self.fc3.bias.detach().numpy()))
            #
            # self.fc4.weight.copy_(torch.from_numpy(M4 @ self.fc4.weight.detach().numpy() @ np.linalg.inv(M3)))
            # self.fc4.bias.copy_(torch.from_numpy(M4 @ self.fc4.bias.detach().numpy()))
            #
            # self.fc5.weight.copy_(torch.from_numpy(M5 @ self.fc5.weight.detach().numpy() @ np.linalg.inv(M4)))
            # self.fc5.bias.copy_(torch.from_numpy(M5 @ self.fc5.bias.detach().numpy()))
            #
            # self.fc6.weight.copy_(torch.from_numpy(M6 @ self.fc6.weight.detach().numpy() @ np.linalg.inv(M5)))
            # self.fc6.bias.copy_(torch.from_numpy(M6 @ self.fc6.bias.detach().numpy()))
            #
            # self.fc7.weight.copy_(torch.from_numpy(M7 @ self.fc7.weight.detach().numpy() @ np.linalg.inv(M6)))
            # self.fc7.bias.copy_(torch.from_numpy(M7 @ self.fc7.bias.detach().numpy()))
            #
            # self.fc8.weight.copy_(torch.from_numpy(M8 @ self.fc8.weight.detach().numpy() @ np.linalg.inv(M7)))
            # self.fc8.bias.copy_(torch.from_numpy(M8 @ self.fc8.bias.detach().numpy()))
            #
            # self.fc9.weight.copy_(torch.from_numpy(self.fc9.weight.detach().numpy() @ np.linalg.inv(M8)))

        # print("updated weights and biases")


    def smallmatrices(self):
        with torch.no_grad():
            U, S, Vh = np.linalg.svd(self.fc2.weight.detach().numpy(), full_matrices=False)
            S *= np.arange(S.size) < 10
            self.fc2.weight.copy_(torch.from_numpy(U @ np.diag(S) @ Vh))

            U, S, Vh = np.linalg.svd(self.fc3.weight.detach().numpy(), full_matrices=False)
            S *= np.arange(S.size) < 10
            self.fc3.weight.copy_(torch.from_numpy(U @ np.diag(S) @ Vh))

    def smallmatrices2(self):
        with torch.no_grad():
            M = self.fc2.weight.detach().numpy()
            col_norms = np.linalg.norm(M, axis=0)
            M[:, col_norms.argsort()[:-10]] = 0

            self.fc2.weight.copy_(torch.from_numpy(M))

            M = self.fc3.weight.detach().numpy()
            col_norms = np.linalg.norm(M, axis=0)
            M[:, col_norms.argsort()[:-10]] = 0

            self.fc3.weight.copy_(torch.from_numpy(M))


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

testset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=False,
    transform=transform
)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


def train(lr, maxepochs, trainlength, variance = 0.5):
    model = MyNet(variance)
    params = list(model.parameters())

    optimizer = optim.SGD(params, lr=lr)
    criterion = nn.NLLLoss()

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)

    time0 = time()
    model.train()

    losses = []

    vallosses = []

    gradnorms = []

    e= 0

    modelold = MyNet(variance)

    val321 = 0

    stopepoch = maxepochs

    np.random.seed(123)

    while e < maxepochs:
        e += 1
        running_loss = 0.0

        total_norm = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            out = model(images)

            loss_ce = criterion(out, labels)

            loss = loss_ce + 0.001*model.fc1.weight.norm(p=1) + 0.001*model.fc2.weight.norm(p=1) + 0.001*model.fc3.weight.norm(p=1)
            optimizer.zero_grad()
            loss.backward()


            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

            # for g in optimizer.param_groups:
            #     if total_norm != 0:
            #         g['lr'] = lr/np.sqrt(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss_ce.item()
        # scheduler.step()
        print(f"{lr}   Epoch {e + 1}/{maxepochs} - Classification Loss: {running_loss / len(trainloader):.4f} - GradNorm: {np.sqrt(total_norm):.3f}")


        losses.append(running_loss / len(trainloader))

        val = validate(model=model, dataset=testset)
        # print(val*100)
        vallosses.append(val*100)

        total_norm = total_norm ** 0.5
        gradnorms.append(total_norm)

        if val > 0.97:
            stopepoch = e
            break

        if e > 15:
            model.smallmatrices2()

        val123 = validate(model=model, dataset=testset)
        print("prejump", val123)


        print(np.linalg.cond(model.fc2.weight.detach().numpy()))
        print(np.linalg.cond(model.fc3.weight.detach().numpy()))



        # if np.isnan(running_loss):
        #     model.load_state_dict(modelold.state_dict())
        #     print(validate(model=model, dataset=testset))
        #     print("reverted back")
        #     for g in optimizer.param_groups:
        #         g['lr'] = g['lr']/10
        # else:
        #     val321 = validate(model=model, dataset=testset)
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr

        if e > 0 and e % trainlength == 0:
            optimizer.zero_grad()
            # plt.imshow(model.fc2.weight.detach().numpy())
            # plt.show()

            # val321 = validate(model=model, dataset=testset)

            # modelold.load_state_dict(model.state_dict())

            model.randomjump()

            plt.imshow(model.fc2.weight.detach().numpy())
            plt.colorbar()
            plt.show()
            plt.imshow(model.fc3.weight.detach().numpy())
            plt.colorbar()
            plt.show()
            plt.imshow(model.fc4.weight.detach().numpy())
            plt.colorbar()
            plt.show()
            for g in optimizer.param_groups:
                g['lr'] = lr
            # vallosses.append(val*100)
            # print("postjump", validate(model=model, dataset=testset))
            optimizer.zero_grad()

    # print(f"\nTraining Time (minutes): {(time() - time0) / 60:.2f}")

    # plt.semilogy(losses)
    # a = trainlength
    # while a < stopepoch:
    #     plt.vlines(x=a, ymin=np.min(losses), ymax=np.max(losses), colors='r', linestyles='--')
    #     a += trainlength
    # plt.title(f"training loss vs epochs - trainperiod:{trainlength}")
    # plt.show()
    #
    # plt.plot(vallosses)
    # a = trainlength
    # while a < stopepoch:
    #     plt.vlines(x=a, ymin=np.min(vallosses), ymax=np.max(vallosses), colors='r', linestyles='--')
    #     a += trainlength
    # plt.title("test accuracy vs epochs")
    # plt.show()
    #
    # plt.plot(gradnorms)
    # a = trainlength
    # while a < stopepoch:
    #     plt.vlines(x=a, ymin=np.min(gradnorms), ymax=np.max(gradnorms), colors='r', linestyles='--')
    #     a += trainlength
    # plt.title("total gradient norm vs epochs")
    # plt.show()

    # plt.semilogy(losses[:10*stopepoch])
    # a = trainlength
    # while a < stopepoch:
    #     plt.vlines(x=a, ymin=np.min(losses), ymax=np.max(losses), colors='r', linestyles='--')
    #     a += trainlength
    # plt.title(f"training loss vs epochs - trainperiod:{trainlength}")
    # plt.show()


    # np.savetxt("weights.txt", model.get_weights()[1].detach().numpy())

    return stopepoch

def validate(model, dataset):
    model.eval()

    correct_count, all_count = 0, 0

    valloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    for images, labels in valloader:
        images = images.view(images.shape[0], -1)

        with torch.no_grad():
            out = model(images)

        ps = torch.exp(out)

        ps_cpu = ps.cpu().numpy()
        labels_cpu = labels.cpu().numpy()

        pred_labels = np.argmax(ps_cpu, axis=1)

        correct_count += (pred_labels == labels_cpu).sum()
        all_count += len(labels_cpu)

    return correct_count / all_count

if __name__ == '__main__':

    print(train(lr=0.1, maxepochs=100, trainlength=1, variance = 0.5))