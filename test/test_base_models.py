from models.mcnn import make_cnn
from models.resnet18k import make_resnet18k
import itertools
import pytest
import torch

# both PN inspired models' instantiators use (c,num_class,ch) as their init signature

test_c=[1,17,71]
test_num_class=[1,10]
test_ch=[1,2,3]

test_pm=[t for t in itertools.product(test_c,test_num_class,test_ch)]

def generate_mock_input(ch):
    # generate cifar sized mock input (32,32)

    return torch.rand((3,ch,32,32))



@pytest.mark.parametrize("c,n_class,ch",test_pm)
def test_init_cnn(c,n_class,ch):
    model=make_cnn(c,n_class,ch)
    out=model(generate_mock_input(ch))

    assert out.shape==(3,n_class)

@pytest.mark.parametrize("c,n_class,ch",test_pm)
def test_init_resnet(c,n_class,ch):
    model=make_resnet18k(c,n_class,ch)
    out=model(generate_mock_input(ch))

    assert out.shape==(3,n_class)


