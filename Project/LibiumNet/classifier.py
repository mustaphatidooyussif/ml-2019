"""
This model generates generator of the datasets for the Network. 

@authors : Mustapha Tidoo Yussif, Samuel Atule, Jean Sabastien Dovonon
         and Nutifafa Amedior. 
"""

from model import LibiumNet
from generator import GenerateDataset
import math 

def main():
    # Display architecture of the network
    model = LibiumNet(output_size=2)
    model.summary()
    """
    NB: 
    The folder structure of our datasets
    /datasets
        /train
         -class1
            -video 1
            -video 2
            - video 3
            - video 4
            - ....
         -class2
           -video 1
            -video 2
            - ..
         ....
       /validation
           /class 1
             -vidoe 1
             - ...
            /class 2
              - video 1
              - ...
              
    """

    # training model
    train_path = "path/to/your/training/dataset/folder"
    num_tain_classes = 'number of classes to train on'
    gen = GenerateDataset(train_path, 'train', num_tain_classes)
    datasets = gen.generator()
    num_samples = gen.get_sample_size()
    steps_per_epoch = math.ceil(num_samples / 2)

    # validation
    val_path = "path/to/your/validation/dataset/folder"
    num_val_classes = 'number of classes to validate on'
    val_gen = GenerateDataset(val_path, 'val', num_val_classes)
    val_datasets = val_gen.generator()
    num_valid_samples = val_gen.get_sample_size()
    steps_per_valid_epoch = math.ceil(num_valid_samples / 2)

    # Train
    model.train(datasets, steps_per_epoch = steps_per_epoch, epochs=25,validation_data=val_datasets, validation_steps=steps_per_valid_epoch)

    # Save model
    path = "path/to/.h5/file/to/save/in"
    model.model.save(path)

if __name__ == "__main__":
    main()