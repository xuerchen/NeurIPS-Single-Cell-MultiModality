import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

class MLP(pl.LightningModule):
    def __init__(self,in_dim,out_dim,config):
        super(MLP, self).__init__()
        H = config.num_hidden
        p = config.dropout
        self.config = config
        self.fc1 = nn.Linear(in_dim, H)
        self.fc2 = nn.Linear(H,H//2)
        self.fc3 = nn.Linear(H//2+H, out_dim)
        self.dp1 = nn.Dropout(p=p)
        self.dp2 = nn.Dropout(p=p)

    def forward(self, x):
        x = self.dp1(x)
        x1 = F.relu(self.fc1(x))
        x1 = self.dp2(x1)
        x = F.relu(self.fc2(x1))
        x = torch.cat([x,x1],dim=1)
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_nb):
        x,y = batch
        yp = self(x)
        criterion = nn.MSELoss()
        loss = criterion(yp.squeeze(), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        yp = self(x)
        criterion = nn.MSELoss()
        loss = criterion(yp.squeeze(), y)
        self.log('valid_RMSE', loss**0.5, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        if len(batch) == 2:
            x,_ = batch
        else:
            x = batch
        return self(x)
    
    def configure_optimizers(self):
        lr = self.config.lr
        wd = float(self.config.wd)
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        
        

    
    