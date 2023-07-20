# import libraries
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

# custom pytorch dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root, file_name_list, train=True, test_size=0.3, random_state=42, transform = None):
        labels = [int(file_name[:-4].split('_')[-1])-1 for file_name in file_name_list]
        full_file_name_list = [root + '/' + file_name for file_name in file_name_list]

        train_data_dir,test_data_dir,train_labels,test_labels = train_test_split(full_file_name_list, labels, test_size=test_size, random_state=random_state)

        self.train = train
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.transform = transform

    def __len__(self):
        if self.train:
            return len(self.train_data_dir)
        else:
            return len(self.test_data_dir)
    
    def __getitem__(self,index):        
        if self.train:
            img_path = self.train_data_dir[index]
            label = self.train_labels[index]
        else:
            img_path = self.test_data_dir[index]
            label = self.test_labels[index]

        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image.to(torch.float32), label

# design a simple cnn model

# training function

# testing function

# main
if __name__ == "__main__":

    root_folder = 'data/data'

    image_filename_list = os.listdir(root_folder)

    train_dataset = CustomImageDataset(
        root=root_folder,
        file_name_list=image_filename_list,
        train=True,
        test_size=0.2
        )
    test_dataset = CustomImageDataset(
        root=root_folder,
        file_name_list=image_filename_list,
        train=False,
        test_size=0.2
        )
    