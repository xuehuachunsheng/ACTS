#-*- encoding:utf-8
import os, sys, json

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class KFPIDataset(Dataset):
    classes_finest = ['Backpacks', 'Belts1', 'Bra', 'Briefs', 'Capris', 'Caps', 'Casual Shoes', 'Clutches', 'Deodorant', 'Dresses', 'Earrings', 'Flats', 'Flip Flops1', 'Formal Shoes', 'Handbags', 'Heels', 'Innerwear Vests', 'Jackets', 'Jeans', 'Kurtas', 'Kurtis', 'Leggings', 'Lipstick', 'Nail Polish', 'Necklace and Chains', 'Nightdress', 'Pendant', 'Perfume and Body Mist', 'Sandals', 'Sarees', 'Shirts', 'Shorts', 'Socks1', 'Sports Shoes', 'Sunglasses', 'Sweaters', 'Sweatshirts', 'Ties1', 'Tops', 'Track Pants', 'Trousers', 'Tshirts', 'Tunics', 'Wallets1', 'Watches1']
    class_mapping_hm = {
        # Level-1
        'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3, 
        # Level-2
        'Bags': 4, 'Belts': 5, 'Bottomwear': 6, 'Dress': 7, 'Eyewear': 8, 'Flip Flops': 9, 'Fragrance': 10, 'Headwear': 11, 'Innerwear': 12, 'Jewellery': 13, 'Lips': 14, 'Loungewear and Nightwear': 15, 'Nails': 16, 'Sandal': 17, 'Saree': 18, 'Shoes': 19, 'Socks': 20, 'Ties': 21, 'Topwear': 22, 'Wallets': 23, 'Watches': 24, 
        # Level-3
        'Backpacks': 25, 'Belts1': 26, 'Bra': 27, 'Briefs': 28, 'Capris': 29, 'Caps': 30, 'Casual Shoes': 31, 'Clutches': 32, 'Deodorant': 33, 'Dresses': 34, 'Earrings': 35, 'Flats': 36, 'Flip Flops1': 37, 'Formal Shoes': 38, 'Handbags': 39, 'Heels': 40, 'Innerwear Vests': 41, 'Jackets': 42, 'Jeans': 43, 'Kurtas': 44, 'Kurtis': 45, 'Leggings': 46, 'Lipstick': 47, 'Nail Polish': 48, 'Necklace and Chains': 49, 'Nightdress': 50, 'Pendant': 51, 'Perfume and Body Mist': 52, 'Sandals': 53, 'Sarees': 54, 'Shirts': 55, 'Shorts': 56, 'Socks1': 57, 'Sports Shoes': 58, 'Sunglasses': 59, 'Sweaters': 60, 'Sweatshirts': 61, 'Ties1': 62, 'Tops': 63, 'Track Pants': 64, 'Trousers': 65, 'Tshirts': 66, 'Tunics': 67, 'Wallets1': 68, 'Watches1': 69
    }
    CLASS_HIERARCHY = [
                       # Level-1
                       -1, -1, -1, -1, 
                       # Level-2
                       0, 0, 1, 1, 0, 2, 3, 0, 1, 0, 3, 1, 3, 2, 1, 2, 0, 0, 1, 0, 0,
                       # Level-3
                       4, 5, 12, 12, 6, 11, 19, 4, 10, 7, 13, 19, 9, 19, 4, 19, 12, 22, 6, 22, 22, 6, 14, 16, 13, 15, 13, 10, 17, 18, 22, 6, 20, 19, 8, 22, 22, 21, 22, 6, 6, 22, 22, 23, 24
                    ]
    C = 70
    def __init__(self, train=True, HLevel=[1,2,3], image_size=224):
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
        self.img_paths, self.labels = KFPIDataset.load_data(train)
        self.transform = transform_train if train else transform_test

    def __getitem__(self, index):
        image_path = self.img_paths[index % len(self.labels)]
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transform(image).float()
        
        label = self.labels[index % len(self.labels)]
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(KFPIDataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = label
            c_name = KFPIDataset.classes_finest[c_id]
            ch_id = KFPIDataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = KFPIDataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = KFPIDataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(45, dtype=np.float32)
                target[label] = 1
            elif self.HLevel == 2:
                target = np.zeros(21, dtype=np.float32)
                c_id = label
                c_name = KFPIDataset.classes_finest[c_id]
                ch_id = KFPIDataset.class_mapping_hm[c_name]
                p_id = KFPIDataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 4] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(4, dtype=np.float32)
                c_id = label
                c_name = KFPIDataset.classes_finest[c_id]
                ch_id = KFPIDataset.class_mapping_hm[c_name]
                p_id = KFPIDataset.CLASS_HIERARCHY[ch_id]
                pp_id = KFPIDataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
                
        target = np.asarray(target, dtype=np.float32)        
        return image, target
    
    def __len__(self):
        return len(self.labels)
    
    def load_data(train):
        images_path = "/home/wyx/datasets/KFPI/FashionProductImages/images/"
        #labels_path = os.path.join("models/KFPI/", train+"_labels.csv")
        train = "train" if train else "test"
        labels_path = "/home/wyx/datasets/KFPI/" + train + "_labels.csv"
        f = open(labels_path, "r")
        f.readline()
        lines = f.readlines()[1:]
        f.close()
        images, labels = [], []
        for line in lines:
            line = line.strip().split(",")
            im_path = os.path.join(images_path, line[0] + ".jpg")
            im_label = KFPIDataset.class_mapping_hm[line[3]] - 25
            images.append(im_path)
            labels.append(im_label)
        images = np.asarray(images)
        labels = np.asarray(labels)
        return images, labels

class KFPIDatasetAL(Dataset):
    
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
    def createLURandomly(nL=900, HLevel=[1,2,3], image_size=224):
        
        images_path = "/home/wyx/datasets/KFPI/FashionProductImages/images/"
        labels_path = "/home/wyx/datasets/KFPI/train_labels.csv"
        f = open(labels_path, "r")
        f.readline()
        lines = f.readlines()[1:]
        f.close()
        C = len(KFPIDataset.classes_finest)
        data = [[] for _ in range(C)] # Rerange by classes
        for line in lines:
            line = line.strip().split(",")
            im_path = os.path.join(images_path, line[0] + ".jpg")
            # Only obtain the finest level
            label = KFPIDataset.class_mapping_hm[line[3]] - 25
            data[int(label)].append((int(label), im_path))
        
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
        L = KFPIDatasetAL(L_data, "L", HLevel, image_size)
        U = KFPIDatasetAL(U_data, "U", HLevel, image_size)
        return L, U

    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,im_path = self.data[datum_id]
        img = Image.open(im_path)
        img = img.convert("RGB")
        img = self.transform(img).float()
        
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(KFPIDataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = label
            c_name = KFPIDataset.classes_finest[c_id]
            ch_id = KFPIDataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = KFPIDataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = KFPIDataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(45, dtype=np.float32)
                target[label] = 1
            elif self.HLevel == 2:
                target = np.zeros(21, dtype=np.float32)
                c_id = label
                c_name = KFPIDataset.classes_finest[c_id]
                ch_id = KFPIDataset.class_mapping_hm[c_name]
                p_id = KFPIDataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 4] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(4, dtype=np.float32)
                c_id = label
                c_name = KFPIDataset.classes_finest[c_id]
                ch_id = KFPIDataset.class_mapping_hm[c_name]
                p_id = KFPIDataset.CLASS_HIERARCHY[ch_id]
                pp_id = KFPIDataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

if __name__ == "__main__":
    from losses import LPSL
    d = KFPIDataset(True,[1,2,3])
    print(d[30000][1])
    