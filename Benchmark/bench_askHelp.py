import numpy as np
import time
import torch
import torch.nn as nn
import numpy as np
import os
from torchsummary import summary
from tqdm import trange

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# %% Dataset
x = np.linspace(0, 10, int(3 * np.power(2, 12))).astype('float32')
y = np.sin(x).astype('float32')


class MyDataset_simple(torch.utils.data.Dataset):
    def __init__(self, x, y, device):
        self.x = torch.tensor(x.reshape(-1, 1)).to(device)
        # self.x = torch.tensor(x.reshape(-1, 1))
        self.y = torch.tensor(y.reshape(-1, 1)).to(device)
        # self.y = torch.tensor(y.reshape(-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


dataset_simple = MyDataset_simple(x, y, device)
my_batch_size = int(np.power(2, 12))
train_dataloader = torch.utils.data.DataLoader(dataset_simple, batch_size=my_batch_size)
# train_dataloader = torch.utils.data.DataLoader(dataset_simple, batch_size=my_batch_size, pin_memory=True)

# %% Dataset2
X = np.linspace(0, 10, int(3 * np.power(2, 12))).astype('float32')
Y = np.sin(X).astype('float32')
X = torch.tensor(X.reshape(-1, 1)).to(device)
Y = torch.tensor(Y.reshape(-1, 1)).to(device)

# %% Model
class MyBenchModelSimple(nn.Module):
    def __init__(self):
        super(MyBenchModelSimple, self).__init__()
        self.my_layer = nn.Sequential(nn.Linear(1, 120),
                                      nn.Tanh(),
                                      nn.Linear(120, 120),
                                      nn.Tanh(),
                                      nn.Linear(120, 120),
                                      nn.Tanh(),
                                      nn.Linear(120, 120),
                                      nn.Tanh(),
                                      nn.Linear(120, 120),
                                      nn.Tanh(),
                                      nn.Linear(120, 1))

    def forward(self, x):
        x = self.my_layer(x)
        return x


model = MyBenchModelSimple().to(device)
summary(model, input_size=(1,))


# %% Training

epoch = 1
my_loss_func = nn.MSELoss()
my_optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

tic = time.perf_counter()
model.train()
while epoch < 1e3 + 1:
    loss_individual_epoch = 0
    for i, (train_input_x, train_input_y) in enumerate(train_dataloader):
        my_optimizer.zero_grad()
        # train_input_x, train_input_y = train_input_x.to(device), train_input_y.to(device)
        outputs = model(train_input_x)

        loss = my_loss_func(outputs, train_input_y)
        loss.backward()
        my_optimizer.step()
        loss_individual_epoch += loss.item()
    loss_individual_epoch = loss_individual_epoch / len(train_dataloader)

    if epoch % 200 == 0:
        toc = time.perf_counter()
        print(f'Epoch: {epoch}, Elapsed: {(toc - tic):.2f} sec')
        tic = time.perf_counter()
        print("Loss for Training on Epoch " + str(epoch) + " is " + str(loss_individual_epoch))

    epoch += 1


# %% Training 2
my_loss_func = nn.MSELoss()
my_optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
model.train()
epochs = int(1e3)
for _ in trange(epochs):
    my_optimizer.zero_grad()
    outputs = model(X)
    loss = my_loss_func(outputs, Y)
    loss.backward()
    my_optimizer.step()

#%% Training 3 (multi threading)
def main():
    train_dataloader = torch.utils.data.DataLoader(dataset_simple, batch_size=my_batch_size, num_workers=2)

if __name__ == '__main__':

