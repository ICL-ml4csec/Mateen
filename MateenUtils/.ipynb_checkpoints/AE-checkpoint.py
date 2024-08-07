import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import random
from tqdm import tqdm


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_shape, int(input_shape*0.75))
        self.fc2 = nn.Linear(int(input_shape*0.75), int(input_shape*0.5))
        self.fc3 = nn.Linear(int(input_shape*0.5), int(input_shape*0.25))
        self.fc4 = nn.Linear(int(input_shape*0.25), int(input_shape*0.1))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(int(input_shape*0.1), int(input_shape*0.25))
        self.fc2 = nn.Linear(int(input_shape*0.25), int(input_shape*0.5))
        self.fc3 = nn.Linear(int(input_shape*0.5), int(input_shape*0.75))
        self.fc4 = nn.Linear(int(input_shape*0.75), input_shape)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def build_autoencoder(input_shape):
    return Autoencoder(input_shape)


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
