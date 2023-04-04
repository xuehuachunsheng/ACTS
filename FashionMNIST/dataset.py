#-*- encoding:utf-8
import os, sys, json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FashionMNISTDataset(Dataset):
    
    # Note: classes_finest id corresponds to the origin data id
    # This sequence order cannot change!
    classes_finest = ["TShirt", "Trouser", "Pullover", "Dress", 
                        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"]
    
    class_mapping_hm = {"Clothes": 0, "Goods": 1, # Level-1
                        'Accessories': 2, 'Bottoms':3, 'Dresses':4, 'Outers':5, 'Shoes':6, 'Tops':7, # Level-2
                         "TShirt":8, "Trouser":9, "Pullover":10, "Dress":11, "Coat":12, "Sandal":13, "Shirt":14, "Sneaker":15, "Bag":16, "AnkleBoot":17 # Level-3
                        }
    CLASS_HIERARCHY = [-1,-1, # corresponding to class_mapping_hm
                       1,0,0,0,1,0,
                       7,3,7,4,5,6,7,6,2,6]
    
    # HLevel: it can be 1,2,3 or [1,2,3]
    # HLevel means the obtained level of labels
    def __init__(self, train=True, HLevel=[1,2,3], image_size=224) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transform = transform_train if train else transform_test
        self.train = train
        self.HLevel = HLevel
        self.load_images()
    
    def load_images(self):
        train_data_path = "/home/wyx/datasets/FashionMNIST/fashion-mnist_train.csv" 
        test_data_path = "/home/wyx/datasets/FashionMNIST/fashion-mnist_test.csv" 
        data_path = train_data_path if self.train else test_data_path
        f = open(data_path, "r")
        raw_data = f.readlines()
        n = len(raw_data) - 1
        f.close()
        images, labels = np.empty(shape=(n, 28, 28),dtype=np.uint8), np.empty(n, dtype=np.int32)
        for i, line in enumerate(raw_data[1:]):
            line = line.strip().split(",")
            labels[i] = int(line[0])
            images[i] = np.reshape([int(x) for x in line[1:]], newshape=(28,28))
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(FashionMNISTDataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = self.labels[index]
            c_name = FashionMNISTDataset.classes_finest[c_id]
            ch_id = FashionMNISTDataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = FashionMNISTDataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = FashionMNISTDataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(10, dtype=np.float32)
                target[self.labels[index]] = 1
            elif self.HLevel == 2:
                target = np.zeros(6, dtype=np.float32)
                c_id = self.labels[index]
                c_name = FashionMNISTDataset.classes_finest[c_id]
                ch_id = FashionMNISTDataset.class_mapping_hm[c_name]
                p_id = FashionMNISTDataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 2] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(2, dtype=np.float32)
                c_id = self.labels[index]
                c_name = FashionMNISTDataset.classes_finest[c_id]
                ch_id = FashionMNISTDataset.class_mapping_hm[c_name]
                p_id = FashionMNISTDataset.CLASS_HIERARCHY[ch_id]
                pp_id = FashionMNISTDataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        return img, target
        
    def __len__(self):
        return len(self.labels)

class FashionMNISTDatasetAL(Dataset):
    
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
    def createLURandomly(data_path="/home/wyx/datasets/FashionMNIST/fashion-mnist_train.csv", nL=500, HLevel=[1,2,3], image_size=224):
        f = open(data_path, "r")
        raw_data = f.readlines()
        f.close()
        data = [[] for _ in range(10)] # Rerange by classes
        for i, line in enumerate(raw_data[1:]):
            line = line.strip().split(",")
            label = int(line[0])
            image = np.reshape([int(x) for x in line[1:]], newshape=(28,28)).astype(np.uint8)
            data[label].append((label, image))

        C = len(FashionMNISTDataset.classes_finest)
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
        L = FashionMNISTDatasetAL(L_data, "L", HLevel, image_size)
        U = FashionMNISTDatasetAL(U_data, "U", HLevel, image_size)
        return L, U

    
    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,img = self.data[datum_id]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(FashionMNISTDataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = label
            c_name = FashionMNISTDataset.classes_finest[c_id]
            ch_id = FashionMNISTDataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = FashionMNISTDataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = FashionMNISTDataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(10, dtype=np.float32)
                target[label] = 1
            elif self.HLevel == 2:
                target = np.zeros(6, dtype=np.float32)
                c_id = label
                c_name = FashionMNISTDataset.classes_finest[c_id]
                ch_id = FashionMNISTDataset.class_mapping_hm[c_name]
                p_id = FashionMNISTDataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 2] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(2, dtype=np.float32)
                c_id = label
                c_name = FashionMNISTDataset.classes_finest[c_id]
                ch_id = FashionMNISTDataset.class_mapping_hm[c_name]
                p_id = FashionMNISTDataset.CLASS_HIERARCHY[ch_id]
                pp_id = FashionMNISTDataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

def test_general():
    d = FashionMNISTDataset(train=True, HLevel=3)
    img,label = d[8529]
    assert list(label) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

def test_CH():
    d = FashionMNISTDataset(train=True, HLevel=[1,2,3])
    img,label = d[8529]
    assert list(label) == [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    
    img,label = d[8536]
    assert list(label) == [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    d = FashionMNISTDataset(train=True, HLevel=2)
    img,label = d[8529]
    assert list(label) == [0, 0, 0, 0, 0, 1]
    
    img,label = d[8536]
    assert list(label) == [1, 0, 0, 0, 0, 0]
    
    d = FashionMNISTDataset(train=True, HLevel=1)
    img,label = d[8529]
    assert list(label) == [0, 1]
    
    img,label = d[8536]
    assert list(label) == [1, 0]

def testAL():
    L, U = FashionMNISTDatasetAL.createLURandomly()
    assert len(L) == 500
    assert len(U) == 59500
    
    ds = set(L.dataidx).union(set(U.dataidx))
    assert len(ds) == len(L) + len(U)

if __name__ == "__main__":
    #test_general()
    #test_CH()
    # testAL()
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from losses import LPSL
    
    HH = LPSL.compute_levels(FashionMNISTDataset.CLASS_HIERARCHY)
    print(HH)
    