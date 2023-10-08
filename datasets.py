import torch
from torch.utils.data import Dataset, DataLoader

class dset(Dataset):
    """dataset to handle reproducible random splitting and transforms"""
    def __init__(self,parent_dataset):
        super().__init__()
        self.parent_dataset=parent_dataset
        self.n=len(self.parent_dataset)
        self.transform=lambda x:x
        self.set_transform_allowed=True

        if self.n==0:
            raise Exception("empty dset cannot be instantiated")
        
    def __len__(self):
        return self.n
    
    def __getitem__(self,i):
        return (self.transform(self.parent_dataset[i][0]),self.parent_dataset[i][1])
        
    def split_random(self,length,seed):
        """
        length is the fractional length of the first part, 0<length<1
        seed is the integer random seed for torch.Generator()
        
        returns two child _dset_s
        """
        
        self.set_transform_allowed=False
        gen=torch.Generator().manual_seed(seed)
        children=torch.utils.data.random_split(self.parent_dataset, [length,1-length], generator=gen)
        
        c=[]
        for child in children:
            _c=dset(child)
            _c.set_transform(self.transform)
            c.append(_c)
            
        return c[0],c[1]
    
    def set_transform(self,fun):
        """
        set transform function, which only takes 1 argument, that is a datapoint
        
        NB: do not include global object references in the function, to prevent unintended transform slip
        """
        if self.set_transform_allowed:
            self.transform=fun
        else:
            raise Exception("Dset has successors, transform change not allowed")
        
    def to_dloader(self,batch_size,shuffle):
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle)