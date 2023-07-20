# import libraries
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

device = "cpu"

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
class CommonRoseNet(nn.Module):
    def __init__(self):
        super(CommonRoseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(X, y)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 30 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# testing function

# main
if __name__ == "__main__":

    root_folder = 'data/data'
    batch_size=64

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
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset)

    model = CommonRoseNet()