import torch.nn as nn
import torch

class IgniteModel(nn.Module):
    def __init__(self,model_class,lr,loss,c,n_class=10,ch=3):
        super().__init__()
        self.model=model_class(c,n_class,ch)
        self.loss=loss
        self.lr=lr
    #    self.save_hyperparameters(ignore=['loss'])
        
    def forward(self,xbatch):
        return self.model(xbatch)
    
    def pretty_loss(self,yhat,y):
        return self.loss(yhat,y).sum()/y.shape[0]
    
    # def training_step(self,batch):
    #     x,y=batch
    #     y_hat=self.forward(x)
    #     loss=self.pretty_loss(y_hat,y)
        
    #     self.log("train loss",loss)
    #     return loss
    
    # def validation_step(self,batch):
    #     x,y=batch
    #     y_hat=self.model(x)
    #     loss=self.pretty_loss(y_hat,y)
        
    #     self.log("valid loss",loss)
        
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer

class ManualModel(nn.Module):
    """
    effective copy of the IgniteModel class to include more involved train and test functions
    NOT meant to be used with the Ignite trainer
    """
    def __init__(self,model_class,lr,loss,c,n_class=10,ch=3):
        super().__init__()
        self.model=model_class(c,n_class,ch)
        self.loss=loss
        self.lr=lr
    #    self.save_hyperparameters(ignore=['loss'])

    def forward(self,xbatch):
        return self.model(xbatch)
    
    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()
        
    
    def pretty_loss(self,yhat,y):
        return self.loss(yhat,y).sum()/y.shape[0]
    
    def get_scalar_loss(self,batch):
        """
        to be called in a location, where the model is set to train/eval modus already
        """
        x,y=batch
        y_hat=self.forward(x)
        
        return self.pretty_loss(y_hat,y)
    

    def valid_loss_standalone(self,batch):
        """
        the standalone verision for the valid loss, preserve loss per datapoint
        """

        self.model.eval()
        x,y=batch
        y_hat=self.forward(x)
        
        return self.loss(y_hat,y)

        
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer

def train_step(model,batch,optimizer,return_loss=False):
    model.set_train()
    optimizer.zero_grad()
    loss=model.get_scalar_loss(batch)
    loss.backward()
    optimizer.step()

    if return_loss:
        return loss

def valid_step(model,batch):
    model.set_eval()

    return model.get_scalar_loss(batch)

        