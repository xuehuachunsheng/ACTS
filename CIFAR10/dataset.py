#-*- encoding:utf-8
import os, sys, json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CIFAR10Dataset(Dataset):
    # 0,transport, animal, sky, water, road, bird, reptile, pet, medium
    classes_finest = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    class_mapping_hm = {"animal": 0, "transport": 1,
                        "bird0": 2, "medium": 3, "pet": 4, "reptile": 5, "road": 6, "sky": 7, "water": 8,
                        "airplane": 9, "automobile": 10, "bird": 11, "cat": 12, "deer": 13, "dog": 14, "frog": 15, "horse": 16, "ship": 17, "truck": 18}
    CLASS_HIERARCHY = [-1, -1,  # Level 1
                       0, 0, 0, 0, 1, 1, 1, # Level 2
                       7, 6, 2, 4, 3, 4, 5, 3, 8, 6 # Level 3
                       ]
    C = 19
    def __init__(self, train=True, HLevel=[1,2,3], image_size=32):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.HLevel = HLevel
        self.train = train
        self.image_size = image_size
        self.mat, self.labels = CIFAR10Dataset.load_data(train)
        self.transform = transform_train if train else transform_test

    def __getitem__(self, index):
        image = self.mat[index % len(self.labels)]
        image = np.transpose(image,(1,2,0))
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image).float()
        
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(CIFAR10Dataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = int(self.labels[index])
            c_name = CIFAR10Dataset.classes_finest[c_id]
            ch_id = CIFAR10Dataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = CIFAR10Dataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = CIFAR10Dataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(10, dtype=np.float32)
                target[int(self.labels[index])] = 1
            elif self.HLevel == 2:
                target = np.zeros(6, dtype=np.float32)
                c_id = int(self.labels[index])
                c_name = CIFAR10Dataset.classes_finest[c_id]
                ch_id = CIFAR10Dataset.class_mapping_hm[c_name]
                p_id = CIFAR10Dataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 2] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(2, dtype=np.float32)
                c_id = int(self.labels[index])
                c_name = CIFAR10Dataset.classes_finest[c_id]
                ch_id = CIFAR10Dataset.class_mapping_hm[c_name]
                p_id = CIFAR10Dataset.CLASS_HIERARCHY[ch_id]
                pp_id = CIFAR10Dataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        
        return image, target
    
    def __len__(self):
        return len(self.labels)
    
    def load_data(train=True):
        def _load(path):
            f = open(path, 'rb')
            dict = pickle.load(f, encoding='bytes')
            images = dict[b'data']
            labels = np.asarray(dict[b'labels'])
            f.close()
            return images, labels
        images = np.zeros([0, 3072])
        labels = np.zeros(0)
        if train:
            for i in range(5):
                path = '/home/wyx/datasets/cifar-10-batches-py/data_batch_{}'.format(i+1)
                t_images, t_labels = _load(path)
                images = np.concatenate((images, t_images),0)
                labels = np.concatenate((labels, t_labels),0)
        else:
            path = '/home/wyx/datasets/cifar-10-batches-py/test_batch'
            images, labels = _load(path)
        images = np.reshape(images, (-1, 3, 32, 32))
        return images, labels


class CIFAR10DatasetAL(Dataset):
    
    # HLevel: it can be 1,2,3 or [1,2,3]
    # HLevel means the obtained level of labels
    # data_type can be one in ["L", "U"]
    # L: initial labeled dataset
    # U: unlabled pool
    def __init__(self, data, data_type="L", HLevel=[1,2,3], image_size=224) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(image_size, image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transform = transform_train if data_type=="L" else transform_test
        self.data = data
        self.dataidx = list(self.data) # Obtains all the datum ids
        self.data_type = data_type
        self.HLevel = HLevel
    
    # Initial Labled Dataset is class balanced
    def createLURandomly(nL=1000, HLevel=[1,2,3], image_size=224):
        
        def _load(path):
            f = open(path, 'rb')
            dict = pickle.load(f, encoding='bytes')
            images = dict[b'data']
            labels = np.asarray(dict[b'labels'])
            f.close()
            return images, labels
        
        images = np.zeros([0, 3072])
        labels = np.zeros(0)
        for i in range(5):
            path = '/home/wyx/datasets/cifar-10-batches-py/data_batch_{}'.format(i+1)
            t_images, t_labels = _load(path)
            images = np.concatenate((images, t_images),0)
            labels = np.concatenate((labels, t_labels),0)
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.transpose(images, (0,2,3,1)).astype(np.uint8)
        data = [[] for _ in range(10)] # Rerange by classes
        for i in range(len(labels)):
            image, label = images[i], labels[i]
            data[int(label)].append((int(label), image))

        C = len(CIFAR10Dataset.classes_finest)
        np.random.seed(0)
        for i in range(C):
            np.random.shuffle(data[i])
        c = nL / C # The number of samples of each class
        L_data = {}
        U_data = {}
        id = 0
        for i in range(C):
            for j in range(len(data[i])):
                if j < c:
                    L_data[str(id)] = data[i][j]
                else:
                    U_data[str(id)] = data[i][j]
                id += 1
        L = CIFAR10DatasetAL(L_data, "L", HLevel, image_size)
        U = CIFAR10DatasetAL(U_data, "U", HLevel, image_size)
        return L, U

    
    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,img = self.data[datum_id]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(CIFAR10Dataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = label
            c_name = CIFAR10Dataset.classes_finest[c_id]
            ch_id = CIFAR10Dataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = CIFAR10Dataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = CIFAR10Dataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(10, dtype=np.float32)
                target[label] = 1
            elif self.HLevel == 2:
                target = np.zeros(7, dtype=np.float32)
                c_id = label
                c_name = CIFAR10Dataset.classes_finest[c_id]
                ch_id = CIFAR10Dataset.class_mapping_hm[c_name]
                p_id = CIFAR10Dataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 2] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(2, dtype=np.float32)
                c_id = label
                c_name = CIFAR10Dataset.classes_finest[c_id]
                ch_id = CIFAR10Dataset.class_mapping_hm[c_name]
                p_id = CIFAR10Dataset.CLASS_HIERARCHY[ch_id]
                pp_id = CIFAR10Dataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

if __name__ == "__main__":
    ds = CIFAR10Dataset(True,[1,2,3],32)
    image, label = ds[10]
    print(label)
    
    