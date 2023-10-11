import torch.nn as nn
import torch


class CNNModel(nn.module):
    def __init__(self, n_filters):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, 
                               stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):    
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
    

# class CNN_LSTM_Model(nn.Module):
#     def __init__(self, hidden_size, num_layers):
          #! TO DO

#         super(CNN_LSTM_Model, self).__init()

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.lstm1 = nn.LSTM(input_size=25, hidden_size=hidden_size, num_layers=num_layers)
#         self.fc1 = nn.Linear(12_800, 256)
#         self.fc2 = nn.Linear(256, 2)

#     def forward(self, x):
#         #! TO DO
#         pass
