import scipy.fftpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from tqdm import tqdm

from time import time

from torch import dtype
from torchvision import datasets, transforms

def laplaceCoeff(n):
    if n == 0:
        return np.ones(1)
    M = []
    for i in range(2*n+1):
        M.append(np.arange(-n, n+1, 1)**i)

    b = np.zeros(2*n+1)
    b[2] = 2
    c = np.linalg.solve(np.array(M), b)
    return c

def pad(x, l, side='l'):
    if side == 'l':
        return np.concatenate((x, np.zeros(l - x.shape[0])), axis=0)
    elif side == 'r':
        return np.concatenate((np.zeros(l - x.shape[0]), x), axis=0)

def rotate(x, l):
    return np.concatenate((x[x.shape[0]-l:x.shape[0]], x[:x.shape[0]-l]))

def nLaplacian(n):
    H = []
    for i in range((n+1)//2):
        H.append(pad(laplaceCoeff(i), n, side='l'))
    for i in range((n+1)//2-1, -1, -1):
        H.append(pad(laplaceCoeff(i), n, side='r'))

    if n % 2 == 1:
        H.pop(n//2)
    return np.array(H)

def nCircularLaplacian(n):
    H = []
    if n % 2 == 1:
        for i in range(n):
            H.append(rotate(laplaceCoeff(n//2), i + n//2 + 1))

    elif n % 2 == 0:
        for i in range(n):
            H.append(rotate(pad(laplaceCoeff(n//2-1), n), i + n//2 + 1))

    return np.array(H)



#############################
# 1. Data Preparation
#############################

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


#############################
# 2. Define Custom Network
#############################

class MyNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

        # torch.nn.init.zeros_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc2.weight)

    def forward(self, x, return_features=False):

        h1 = F.relu(self.fc1(x))  # shape: (batch_size, hidden1)

        h2 = F.relu(self.fc2(h1))  # shape: (batch_size, hidden2)

        out = F.log_softmax(self.fc3(h2), dim=1)  # shape: (batch_size, output_size)

        if return_features:
            return h1, h2, out, self.fc1.weight, self.fc2.weight, self.fc3.weight
        else:
            return out

    def get_weights(self):
        return self.fc1.weight, self.fc2.weight, self.fc3.weight





#############################
# 3. Instantiate Model & Parameters
#############################
def testalpha(alpha, lr):
    # Dimensions
    input_size = 784  # 28*28
    hidden1_size = 16
    hidden2_size = 8
    output_size = 10

    model = MyNet(input_size, hidden1_size, hidden2_size, output_size)


    # HPF1 = np.diag(np.arange(hidden1_size)/hidden1_size) @ sp.linalg.dft(hidden1_size)
    # HPF2 = np.diag(np.arange(hidden2_size)/hidden2_size) @ sp.linalg.dft(hidden2_size)

    # HPF1 = sp.linalg.inv(sp.linalg.dft(hidden1_size, scale='sqrtn')) @ np.diag(np.arange(hidden1_size)/hidden1_size) @ sp.linalg.dft(hidden1_size, scale='sqrtn')
    # HPF2 = sp.linalg.inv(sp.linalg.dft(hidden2_size, scale='sqrtn')) @ np.diag(np.arange(hidden2_size)/hidden2_size) @ sp.linalg.dft(hidden2_size, scale='sqrtn')
    # HPF3 = np.diag(np.arange(output_size)/output_size > 0.2) @ sp.linalg.dft(output_size)


    # HAM1 = np.diag(np.arange(1, hidden1_size+1)/1.0)
    # HAM1 -= np.diag(np.arange(1, hidden1_size)/2,1)
    # HAM1 -= np.diag(np.arange(2, hidden1_size+1) / 2, -1)
    #
    # HAM2 = np.diag(np.arange(1, hidden2_size+1)/1.0)
    # HAM2 -= np.diag(np.arange(1, hidden2_size) / 2, 1)
    # HAM2 -= np.diag(np.arange(2, hidden2_size+1) / 2, -1)
    #
    # M1 = torch.from_numpy(HAM1.astype(np.float32))
    # M2 = torch.from_numpy(HAM2.astype(np.float32))

    # M1 = torch.from_numpy(np.real(HPF1).astype(np.float32))
    # M2 = torch.from_numpy(np.real(HPF2).astype(np.float32))
    #M3 = torch.from_numpy(np.real(HPF3).astype(np.float32))

    M1 = torch.from_numpy(nCircularLaplacian(hidden1_size).astype(np.float32))
    M2 = torch.from_numpy(nCircularLaplacian(hidden2_size).astype(np.float32))


    #############################
    # 4. Move to MPS (if available) or CPU
    #############################
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    M1 = M1.to(device)
    M2 = M2.to(device)
    #M3 = M3.to(device)





    #############################
    # 5. Define Optimizer & Loss
    #############################

    params = list(model.parameters())

    optimizer = optim.SGD(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    criterion = nn.NLLLoss()


    #############################
    # 6. Training Loop
    #############################

    reg = 5
    trainrep = 45
    number = 10

    time0 = time()
    model.train()

    losses = []
    conds = []

    for e in range(number * (trainrep+reg)):
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            h1, h2, out, w1, w2, w3 = model(images, return_features=True)

            loss_ce = criterion(out, labels)

            #print(h1.shape)
            #print(M1.shape)
            # loss_h1 = torch.matmul(h1, M1).norm(p=1)
            # loss_h2 = torch.matmul(h2, M2).norm(p=1)
            # loss_out = torch.matmul(out, M3).norm()

            # loss_regularization = loss_h1 + loss_h2# + loss_out

            z1 = torch.matmul(h1, M1)
            z2 = torch.matmul(h2, M2)

            #loss_regularization = 10 * z1.norm(p=2) / (h1-z1).norm(p=2) + z2.norm(p=2) / (h2-z2).norm(p=2) + 10*abs(h1.mean()) + abs(h2.mean())
            #loss_regularization = 100 * torch.log(z1.norm(p=2)) - 100 * torch.log((h1 - z1).norm(p=2)) + torch.log(z2.norm(p=2)) - torch.log((h2 - z2).norm(p=2)) + 100 * abs(h1.mean()) + abs(h2.mean())
            loss_regularization = z1.norm(p=2) + z2.norm(p=2)# + (w1.norm() + w2.norm())

            loss = loss_ce + alpha*loss_regularization

            # Total loss
            # if e % (reg + trainrep) < reg:
            #     loss = alpha * loss_regularization
            #     # for g in optimizer.param_groups:
            #     #     g['lr'] = lr*pow(10, (10 - e//(reg+trainrep))/5)
            # else:
            #     loss = loss_ce + alpha * loss_regularization
            #     # for g in optimizer.param_groups:
            #     #     g['lr'] = lr

            loss.backward()
            optimizer.step()

            running_loss += loss_ce.item()

        print(f"  {alpha}   {lr}  ({e % (reg + trainrep) < reg})   Epoch {e + 1}/{number * (trainrep+reg)} - Classification Loss: {running_loss / len(trainloader):.4f} - Filter Loss: {loss_regularization} - CondNumb: {np.linalg.cond(w1.detach().numpy())}")


        plt.plot(range(hidden1_size), np.mean(h1.detach().numpy(), axis=0), 'r')
        plt.plot(range(hidden2_size), np.mean(h2.detach().numpy(), axis=0), 'b')
        plt.title(f"Alpha: {alpha}      lr = {optimizer.param_groups[0]['lr']}")
        plt.show()

        plt.imshow(model.get_weights()[2].detach().numpy())
        plt.title(f"Alpha: {alpha}      lr = {optimizer.param_groups[0]['lr']}")
        plt.show()

        plt.imshow(model.get_weights()[1].detach().numpy())
        plt.title(f"Alpha: {alpha}      lr = {optimizer.param_groups[0]['lr']}")
        plt.show()


        losses.append(running_loss / len(trainloader))
        conds.append(np.linalg.cond(w1.detach().numpy()))

        if e % 20:
            val = validate(model=model, dataset=valset, device=device)
            if val > 0.94:
                break

        scheduler.step()

    print(f"\nTraining Time (minutes): {(time() - time0) / 60:.2f}")

    plt.semilogy(losses)
    plt.title("classification loss vs epochs")
    plt.show()

    plt.plot(conds)
    plt.title("CondNumb vs epochs")
    plt.show()


    return validate(model=model, dataset=testset, device=device)

def validate(model, dataset, device):
    model.eval()

    correct_count, all_count = 0, 0

    valloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    for images, labels in valloader:
        images, labels = images.to(device), labels.to(device)

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

if __name__ == "__main__":

    print(testalpha(0.0002, 0.04))

    # L = nLaplacian(100)
    #
    # print(L)
    #
    # plt.imshow(L)
    # plt.show()
