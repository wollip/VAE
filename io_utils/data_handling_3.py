from torch.utils.data import Dataset
import h5py

import numpy as np

class WCH5Dataset(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded 
    """


    def __init__(self, path, train_indices_file, validation_indices_file, test_indices_file, shuffle=1, 
                 transform=None, reduced_dataset_size=None, seed=42):


        f=h5py.File(path,'r')
        hdf5_event_data = f["event_data"]
        hdf5_labels=f["labels"]
        hdf5_energies=f["energies"]
        hdf5_positions=f["positions"]
        hdf5_angles=f["angles"]

        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

        event_data_shape = hdf5_event_data.shape
        event_data_offset = hdf5_event_data.id.get_offset()
        event_data_dtype = hdf5_event_data.dtype

        #this creates a memory map - i.e. events are not loaded in memory here
        #only on get_item
        self.event_data = np.memmap(path, mode='r', shape=event_data_shape, offset=event_data_offset, dtype=event_data_dtype)
        
        #this will fit easily in memory even for huge datasets
        self.labels = np.array(hdf5_labels)
        
        # The energies will also fit easily in memory
        self.energies = np.array(hdf5_energies)
        
        # The positions will also fit easily in memory
        self.positions = np.array(hdf5_positions)
        
        # The angles will also fit easily in memory
        self.angles = np.array(hdf5_angles)
        
        self.transform=transform

        # Setup the indices to be used for different sections of the dataset
        self.train_indices = self.load_indicies(train_indices_file)
        self.val_indices = self.load_indicies(validation_indices_file)
        self.test_indices = self.load_indicies(test_indices_file)
         
        n_val = len(self.val_indices)
        n_test = len(self.test_indices)
                
    def load_indicies(self, indicies_file):
        with open(indicies_file, 'r') as f:
            lines = f.readlines()
        # indicies = [int(l.strip()) for l in lines if not l.isspace()]
        indicies = [int(l.strip()) for l in lines]
        return indicies

    def __getitem__(self,index):
        if self.transform is None:
            return np.array(self.event_data[index,:]),  self.labels[index], self.energies[index], self.angles[index],  index, self.positions[index]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.labels.shape[0]
