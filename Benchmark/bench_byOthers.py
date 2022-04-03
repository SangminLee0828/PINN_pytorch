import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import math
from tqdm import trange
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchSize = 10000
epochs = 5000

model = nn.Sequential(
    nn.Linear(1, 50, device='cuda'),
    nn.ReLU(),
    nn.Linear(50, 50, device='cuda'),
    nn.ReLU(),
    nn.Linear(50, 50, device='cuda'),
    nn.ReLU(),
    nn.Linear(50, 50, device='cuda'),
    nn.ReLU(),
    nn.Linear(50, 1, device='cuda')
)

summary(model, input_size=(1,))

#%%
class MyBenchModelSimple(nn.Module):
    def __init__(self):
        super(MyBenchModelSimple, self).__init__()
        self.my_layer = nn.Sequential(nn.Linear(1, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 1))

    def forward(self, x):
        #You code after this line
        x = self.my_layer(x)
        return x

model = MyBenchModelSimple().to(device)
summary(model, input_size=(1,))
#%%
opt = torch.optim.Adam(params=model.parameters(), lr=1e-2)

X = torch.linspace(-5.0, 5.0, batchSize, device='cuda').reshape(-1,1)
Y = (1/(math.sqrt(2*math.pi)))*torch.exp(-X**2/2)

X = np.linspace(0, 10, int(3 * np.power(2, 12))).astype('float32')
Y = np.sin(X).astype('float32')
X = torch.tensor(X.reshape(-1, 1)).to(device)
Y = torch.tensor(Y.reshape(-1, 1)).to(device)


loss = torch.nn.MSELoss()
for _ in trange(epochs):
    opt.zero_grad()
    Ypred = model(X)
    lossVal = loss(Ypred, Y)
    lossVal.backward()
    opt.step()

# plt.plot(X.detach().cpu(), Ypred.detach().cpu())

#%%
X.size()
