import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
from tqdm import tqdm


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

class autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75), int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size*0.1),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75),int(feature_size)),
                                     )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


def train_autoencoder(model, train_loader, num_epochs=100, learning_rate=0.0001):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)          
    for epoch in tqdm(range(num_epochs)):
        model.train() 
        for batch_data in train_loader:
            inputs = batch_data[0].to(device).float()
            targets = batch_data[0].to(device).float()
            optimizer.zero_grad()   
            outputs = model(inputs)
            loss = criterion(outputs, targets)  
            loss.backward()  
            optimizer.step()          
        model.eval()
    return model
