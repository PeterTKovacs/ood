from custom_models import train_step, valid_step
from torch.utils.data import DataLoader
import torch

def train_and_measure_loop(model,optimizer,train_loader,ds_measure,measurement_interval=1,return_loss=False):
    """
    generic train loop
    we assume that model is already sent to device
    """

    loss=[]
    measurements=[]

    m_dl=DataLoader(ds_measure,batch_size=len(ds_measure),shuffle=False)
    full_meas_batch=next(iter(m_dl))

    for i,batch in enumerate(iter(train_loader)):
        if return_loss:
            loss.append(train_step(model,batch,optimizer,return_loss))
        else:
            train_step(model,batch,optimizer,return_loss)

        if i%measurement_interval==0:
            measurements.append((i,model.valid_loss_standalone(full_meas_batch)))

    return measurements,loss

def valid_loop(model,val_loader):

    """
    generic valid loop
    we assume that model is already sent to device
    """

    avg_loss=0
    n=0
    with torch.no_grad():
        for batch in iter(val_loader):
            avg_loss+=valid_step(model,batch)*batch[0].shape[0]
            n+=batch[0].shape[0]

    return avg_loss/n