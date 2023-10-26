import torch.nn as nn
import torch

def calc_shape(in_shape, padding, dilation, k_size, stride):
    h = (in_shape[0] + 2*padding - dilation * (k_size-1) - 1) / stride + 1
    w = (in_shape[1] + 2*padding - dilation * (k_size-1) - 1) / stride + 1
    return (int(h), int(w))

class CNNModel(nn.Module):
    def __init__(self, n_filters, input_shape):
        """
        """
        super(CNNModel, self).__init__()

        # In one channel, out n_filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, 
                               stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        # Calculate next layer input size
        conv_out = calc_shape([input_shape[0], input_shape[1]], 0, 1, 3, 1)
        maxp_out = calc_shape([conv_out[0], conv_out[1]], 0, 1, 2, 1)

        self.fc1 = nn.Linear(maxp_out[0]*maxp_out[1]*n_filters, 256)
        self.fc2 = nn.Linear(256, 2)

        # Not needed
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):    
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        x = self.fc2(x)

        # Not needed
        # x = self.softmax(x)

        return x
    

class CNN_LSTM_Model(nn.Module):
     def __init__(self, hidden_size, num_layers):
         super(CNN_LSTM_Model, self).__init__()

         self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3, stride=1, padding=1)
         self.relu = nn.ReLU()
         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
         self.lstm1 = nn.LSTM(input_size=25, hidden_size=hidden_size, num_layers=num_layers)
         self.fc1 = nn.Linear(12_800, 256)
         self.fc2 = nn.Linear(256, 2)

     def forward(self, x):
         x = self.conv1(x)
         x = self.relu(x)
         x = self.maxpool(x)

        # Flatten
         x = torch.flatten(x, 1)
         x = self.lstm1(x)

         x = self.fc1(x)

         x = self.fc2(x)
         x = self.softmax(x)

         return x
