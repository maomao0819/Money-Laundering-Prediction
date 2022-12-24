import torch
import torch.nn as nn

class DNN_Model(nn.Module):
    def __init__(self, n_feature=26):
        super(DNN_Model, self).__init__()
        self.fc_reconstruct = nn.Sequential(
            nn.Linear(n_feature, 128),
            nn.Mish(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(32, n_feature),
            nn.Mish(True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_feature, 16),
            nn.Mish(True),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, input):
        feature = self.fc_reconstruct(input)
        output = self.classifier(feature)
        return output

class DNN_Model_Prob(DNN_Model):
    def __init__(self, n_feature=26):
        super().__init__(n_feature=n_feature)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feature = self.fc_reconstruct(input)
        output = self.classifier(feature)
        prob = self.sigmoid(output)
        return prob