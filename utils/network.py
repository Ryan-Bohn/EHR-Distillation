import torch
import torch.nn as nn
import torch.nn.functional as F

class IHMPreliminaryMLP(nn.Module):
    def __init__(self, input_shape=(48, 59), init_distr=None):
        super(IHMPreliminaryMLP, self).__init__()
        
        self.fc1 = nn.Linear(in_features=input_shape[0]*input_shape[1], out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)
        if init_distr == "kaiming":
            self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input tensor row-firstly, first dim is batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class IHMPreliminary1DCNN(nn.Module):
    def __init__(self, input_shape=(48, 59), init_distr=None):
        """
        init_distr: None, "kaiming", or "xavier"
        """
        super(IHMPreliminary1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=input_shape[0]*128, out_features=256)  # Adjusted to match the input dimensions
        self.fc2 = nn.Linear(in_features=256, out_features=2)  # Output 2-dim scalar for binary classification
        if init_distr == "kaiming":
            self.kaiming_init()
    
    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x.permute(0, 2, 1))) # Transpose (seqlen * num_features) into channel-first
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here as CrossEntropyLoss applies softmax internally
        return x