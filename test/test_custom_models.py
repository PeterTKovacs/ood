from models.custom_models import IgniteModel, ManualModel
from models.mcnn import make_cnn
from models.resnet18k import make_resnet18k
import itertools
import pytest
import torch
import torch.nn as nn

def generate_mock_batch(n_batch,n_class=10,ch=3):
    # generate a mock batch of n (32,32) images of ch channels

    x=torch.rand(n_batch,ch,32,32)
    y=torch.randint(0,n_class,(n_batch,))

    return (x,y)

def supposed_loss(y_hat,y):
    _exp=torch.exp(y_hat)
    return -torch.log(_exp[y]/_exp.sum())

def loss_equal(l1,l2):
    # sets equality criterion for floating loss tensors

    diff=torch.abs(l1-l2)
    sum_abs=torch.abs(l1)+torch.abs(l2)

    try:
        return diff/sum_abs
    except:
        return True

test_c=[1,17,71]
test_num_class=[1,10]
test_ch=[1,2,3]

test_pm=[t for t in itertools.product(test_c,test_num_class,test_ch)]

@pytest.mark.parametrize("c,n_class,ch",test_pm)
def test_init_cnn(c,n_class,ch):
    model=IgniteModel(make_cnn,1e-4,nn.CrossEntropyLoss(reduction='none'),c,n_class,ch)
    out=model(generate_mock_batch(1,n_class,ch)[0])

    assert out.shape==(1,n_class)

@pytest.mark.parametrize("c,n_class,ch",test_pm)
def test_init_resnet(c,n_class,ch):
    model=IgniteModel(make_resnet18k,1e-4,nn.CrossEntropyLoss(reduction='none'),c,n_class,ch)
    out=model(generate_mock_batch(5,n_class,ch)[0])

    assert out.shape==(5,n_class)

test_cases_loss=[(make_cnn,2,1),(make_resnet18k,2,1),(make_cnn,10,10),(make_resnet18k,2,10),(make_cnn,2,15)]

@pytest.mark.parametrize('model_fun,n_class,n_batch',test_cases_loss)
def test_scalar_loss(model_fun,n_class,n_batch):
    model=ManualModel(model_fun,1e-4,nn.CrossEntropyLoss(reduction='none'),10,n_class,3)
    x,y=generate_mock_batch(n_batch,n_class,3)
    y_hat=model(x)

    sum_loss=0
    for i in range(n_batch):
        sum_loss+=supposed_loss(y_hat[i],y[i])
    
    assert loss_equal(sum_loss/n_batch,model.get_scalar_loss((x,y)))<1e-3

@pytest.mark.parametrize('model_fun,n_class,n_batch',test_cases_loss)
def test_standalone_loss(model_fun,n_class,n_batch):
    model=ManualModel(model_fun,1e-4,nn.CrossEntropyLoss(reduction='none'),10,n_class,3)
    x,y=generate_mock_batch(n_batch,n_class,3)
    y_hat=model(x)

    yraw=model.valid_loss_standalone((x,y))

    for i in range(n_batch):
        assert loss_equal(supposed_loss(y_hat[i],y[i]),yraw[i])


