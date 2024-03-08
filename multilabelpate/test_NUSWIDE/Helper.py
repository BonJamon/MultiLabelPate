#Pipeline Helper
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
#from sklearn.metrics import precision_score, recall_score, f1_score
import json
from shutil import copyfile
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from collections import OrderedDict
import sys
import torch.nn as nn
from CNN_Networks import Dummy, Resnext50
import pickle


# Simple dataloader and label binarization, that is converting test labels into binary arrays of length 27 (number of classes) with 1 in places of applicable labels).
class NusDataset(Dataset):
    
####################################################################
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_folder = os.path.join('archive','images')
####################################################################

    def __init__(self, data_path, anno_path, transforms):
        self.transforms = transforms
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        self.classes = json_data['labels']

        self.imgs = []
        self.annos = []
        self.data_path = data_path
        #print('loading', anno_path)
        i=0
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
            i+=1
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)

    def preprocessing():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_folder = os.path.join('archive','images')
        label_folder = 'labels'
        # Test preprocessing
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
            #print(tuple(np.array(np.array(mean)*255).tolist()))

        # Train preprocessing
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
                                    shear=None,
                                    fill=tuple(np.array(np.array(mean)*255).astype(int).tolist())),
            #Problem: At some Point in processing the private dataset this throws an error that here should be 3 channel
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])


        # Initialize the dataloaders for training.
        test_annotations = os.path.join(label_folder, 'test_new.json')
        train_annotations = os.path.join(label_folder, 'train_new.json')


        test_data = NusDataset(img_folder, test_annotations, val_transform)
        train_data = NusDataset(img_folder, train_annotations, train_transform)

        return test_data, train_data
        
    def get_teacher_loaders(train_data, num_teachers, batch_size):
        """ Function to create data loaders for the Teacher classifier """
        teacher_loaders = []
        data_size = len(train_data) // num_teachers

        for i in range(num_teachers):
            indices = list(range(i*data_size, (i+1)*data_size))
            subset_data = Subset(train_data, indices)
            loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size)
            teacher_loaders.append(loader)

        return teacher_loaders

class Helperclass:
              
    # Here is an auxiliary function for checkpoint saving.
    def checkpoint_save(model, save_path, epoch):
        f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
        if 'module' in dir(model):
            torch.save(model.module.state_dict(), f)
        else:
            torch.save(model.state_dict(), f)
        #print('saved checkpoint:', f)
    
            # Here is an auxiliary function for checkpoint saving.
    def checkpoint_save_latest(model, save_path):
        f = os.path.join(save_path, 'latest_checkpoint.pth')
        if 'module' in dir(model):
            torch.save(model.module.state_dict(), f)
        else:
            torch.save(model.state_dict(), f)
        #print('saved checkpoint:', f)
        
    def checkpoint_load(model, save_path, epoch):
        f = os.path.join(save_path,'checkpoint-{:06d}.pth'.format(epoch))
        model.load_state_dict(torch.load(f))
        
    def save_models(models, name):
    #saving models
        PATH = "models/"

        dic = dict()
        for i in range(len(models)):
            temp = {'model'+str(i)+'_state_dict': models[i].state_dict()}
            dic.update(temp)
        torch.save(dic, os.path.join(PATH,name))
        
    def load_models(PATH, no_classes):
        checkpoint = torch.load(os.path.join("models",PATH), map_location=torch.device("cpu"))

        #loading models
        models = []
        for i in range(len(checkpoint)):
            model = Resnext50(no_classes)
            #Problem: Generates keys module.base_model instead of base_model
            state_dict = checkpoint['model'+str(i)+'_state_dict']
            new_state_dict = OrderedDict()
            for k,v in state_dict.items():
                name = k.replace("module.","")
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict)
            models.append(model)
        return models 

    def predict(model, dataloader, device):
        outputs = torch.Tensor([])
        outputs = outputs.to(device)
        sigm = nn.Sigmoid()
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                output = sigm(output)
                #ps = torch.argmax(torch.exp(output), dim=1)
                outputs = torch.cat((outputs, output))      
        return outputs.to("cpu")

    def get_predictions(num_teachers):
        with open("results/predictions.p", 'rb') as f:
            loaded_dict = pickle.load(f)
        if num_teachers==2:
            preds = loaded_dict[1]["2 teachers"]
        if num_teachers == 4:
            preds = loaded_dict[2]["4 teachers"]
        if num_teachers == 8:
            preds = loaded_dict[3]["8 teachers"]
        if num_teachers == 16:
            preds = loaded_dict[4]["16 teachers"]
        return preds
    
    def get_labels():
        with open("results/predictions.p", 'rb') as f:
            loaded_dict = pickle.load(f)
        labels = loaded_dict[0]["labels"]
        return labels