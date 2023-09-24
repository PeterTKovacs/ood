import torch.nn as nn
import lightning.pytorch as pl
import torch

class LitModel(pl.LightningModule):
    def __init__(self,model_class,lr,loss,c,n_class=10,ch=3):
        super().__init__()
        self.model=model_class(c,n_class,ch)
        self.loss=loss
        self.lr=lr
        self.save_hyperparameters(ignore=['loss'])
        
    def forward(self,xbatch):
        return self.model(xbatch)
    
    def pretty_loss(self,yhat,y):
        return self.loss(yhat,y).sum()/y.shape[0]
    
    def training_step(self,batch):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.pretty_loss(y_hat,y)
        
        self.log("train loss",loss)
        return loss
    
    def validation_step(self,batch):
        x,y=batch
        y_hat=self.model(x)
        loss=self.pretty_loss(y_hat,y)
        
        self.log("valid loss",loss)
        
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer