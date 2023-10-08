import pytest
from datasets import dset
import random

# test child sizes and seed reproducibility
# test dloader batch size (shuffle?)
# len, getitem, settransform

def generate_sample_data(n,n_class):
    # we work with annotated datasets
    x=random.sample([i for i in range(n)],n)
    y=[random.sample(range(n_class),1)[0] for i in range(n)]

    return [(xx,yy) for xx,yy in zip(x,y)]

test_ds_configs=[(10,2),(1,12),(1242,32)]

@pytest.mark.parametrize("ds_config",test_ds_configs)
def test_len_getitem(ds_config):
    ds=generate_sample_data(*ds_config)
    dataset=dset(ds)

    assert len(dataset)==ds_config[0]
    k=random.sample(range(ds_config[0]),1)[0]
    assert dataset[k][0]==ds[k][0] and dataset[k][1]==ds[k][1]

def test_zero_ds_size(ds_config=(1,2)):
    ds=generate_sample_data(*ds_config)
    dataset=dset(ds)

    with pytest.raises(Exception) as ex:  
        d1,d2=dataset.split_random(0.1,137)
    assert str(ex.value) == "empty dset cannot be instantiated" 


def test_settransform(ds_config=(1000,100)):
    ds=generate_sample_data(*ds_config)
    dataset=dset(ds)
    dataset.set_transform(lambda x: -x)

    assert dataset[0][0]==-1*ds[0][0]

    d1,d2=dataset.split_random(0.5,0)

    assert dataset.set_transform_allowed==False 
    assert d1.set_transform_allowed and d2.set_transform_allowed

def test_split(ds_config=(1000,100)):
    ds=generate_sample_data(*ds_config)
    dataset=dset(ds)

    d1,d2=dataset.split_random(0.3,1)

    assert len(d1)==300 and len(d2)==700

    e1,e2=dataset.split_random(0.3,1)

    assert d1[0][0]==e1[0][0] and d2[4][1]==e2[4][1]

def test_dloader_size(ds_config=(1000,100),batch_size=37,shuffle=True):
    ds=generate_sample_data(*ds_config)
    dataset=dset(ds)

    dl=dataset.to_dloader(batch_size,shuffle)

    batch=next(iter(dl))

    assert batch[0].shape[0]==batch_size and batch[1].shape[0]==batch_size

