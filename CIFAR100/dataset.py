#-*- encoding:utf-8
import os, sys, json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CIFAR100Dataset(Dataset):
    classes_finest = ["Apples", "Aquarium fish", "Baby", "Bear", "Beaver", 
                      "Bed", "Bee", "Beetle", "Bicycle", "Bottles", 
                      "Bowls", "Boy", "Bridge", "Bus", "Butterfly", 
                      "Camel", "Cans", "Castle", "Caterpillar", "Cattle", 
                      "Chair", "Chimpanzee", "Clock", "Cloud", "Cockroach", 
                      "Couch", "Crab", "Crocodile", "Cups", "Dinosaur", 
                      "Dolphin", "Elephant", "Flatfish", "Forest", "Fox", 
                      "Girl", "Hamster", "House", "Kangaroo", "Keyboard", 
                      "Lamp", "Lawn-mower", "Leopard", "Lion", "Lizard", 
                      "Lobster", "Man", "Maple", "Motorcycle", "Mountain", 
                      "Mouse", "Mushrooms", "Oak", "Oranges", "Orchids", 
                      "Otter", "Palm", "Pears", "Pickup truck", "Pine", 
                      "Plain", "Plates", "Poppies", "Porcupine", "Possum", 
                      "Rabbit", "Raccoon", "Ray", "Road", "Rocket", 
                      "Roses", "Sea", "Seal", "Shark", "Shrew", 
                      "Skunk", "Skyscraper", "Snail", "Snake", "Spider", 
                      "Squirrel", "Streetcar", "Sunflowers", "Sweet peppers", "Table", 
                      "Tank", "Telephone", "Television", "Tiger", "Tractor", 
                      "Train", "Trout", "Tulips", "Turtle", "Wardrobe", 
                      "Whale", "Willow", "Wolf", "Woman", "Worm"]
    
    class_mapping_hm = {
        # Level-1
        "Aquatic animals": 0, "Plants": 1, "Household supplies": 2, "Invertebrate": 3, 
        "Vertebrates": 4, "Outdoor things": 5, "People": 6, "Vehicles": 7,
        # Level-2
        "Aquatic mammals": 8, "Fish":9, "Flowers": 10, "Food containers": 11, "Fruit and vegetables": 12, 
        "Household electrical devices": 13, "Household furniture": 14, "Insects": 15, "Large carnivores": 16,
        "Large man-made outdoor things": 17, "Large natural outdoor scenes": 18, 
        "Large omnivores and herbivores": 19, "Medium-sized mammals": 20, 
        "Non-insect invertebrates": 21, "People": 22, "Reptiles": 23, "Small mammals": 24,
        "Trees": 25, "Vehicles1": 26, "Vehicles2":27,
        # Level-3
        "Apples": 28, "Aquarium fish": 29, "Baby": 30, "Bear": 31, "Beaver": 32, 
        "Bed": 33, "Bee": 34, "Beetle": 35, "Bicycle": 36, "Bottles": 37, 
        "Bowls": 38, "Boy": 39, "Bridge": 40, "Bus": 41, "Butterfly": 42, 
        "Camel": 43, "Cans": 44, "Castle": 45, "Caterpillar": 46, "Cattle": 47, 
        "Chair": 48, "Chimpanzee": 49, "Clock": 50, "Cloud": 51, "Cockroach": 52, 
        "Couch": 53, "Crab": 54, "Crocodile": 55, "Cups": 56, "Dinosaur": 57, 
        "Dolphin": 58, "Elephant": 59, "Flatfish": 60, "Forest": 61, "Fox": 62, 
        "Girl": 63, "Hamster": 64, "House": 65, "Kangaroo": 66, "Keyboard": 67, 
        "Lamp": 68, "Lawn-mower": 69, "Leopard": 70, "Lion": 71, "Lizard": 72, 
        "Lobster": 73, "Man": 74, "Maple": 75, "Motorcycle": 76, "Mountain": 77, 
        "Mouse": 78, "Mushrooms": 79, "Oak": 80, "Oranges": 81, "Orchids": 82, 
        "Otter": 83, "Palm": 84, "Pears": 85, "Pickup truck": 86, "Pine": 87, 
        "Plain": 88, "Plates": 89, "Poppies": 90, "Porcupine": 91, "Possum": 92, 
        "Rabbit": 93, "Raccoon": 94, "Ray": 95, "Road": 96, "Rocket": 97, 
        "Roses": 98, "Sea": 99, "Seal": 100, "Shark": 101, "Shrew": 102, 
        "Skunk": 103, "Skyscraper": 104, "Snail": 105, "Snake": 106, "Spider": 107, 
        "Squirrel": 108, "Streetcar": 109, "Sunflowers": 110, "Sweet peppers": 111, "Table": 112, 
        "Tank": 113, "Telephone": 114, "Television": 115, "Tiger": 116, "Tractor": 117, 
        "Train": 118, "Trout": 119, "Tulips": 120, "Turtle": 121, "Wardrobe": 122, 
        "Whale": 123, "Willow": 124, "Wolf": 125, "Woman": 126, "Worm": 127
        }
    
    CLASS_HIERARCHY = [-1,-1,-1,-1,-1,-1,-1,-1, # Level 1
                      # Level 2
                       0,0,1,2,1,2,2,3,4,5,
                       5,4,4,3,6,4,4,1,7,7,
                       # Level 3
                       12, 9, 22, 16, 8, 14, 15, 15, 26, 
                       11, 11, 22, 17, 26, 15, 19, 11, 
                       17, 15, 19, 14, 19, 13, 18, 15, 
                       14, 21, 23, 11, 23, 8, 19, 9, 
                       18, 20, 22, 24, 17, 19, 13, 13, 
                       27, 16, 16, 23, 21, 22, 25, 26, 
                       18, 24, 12, 25, 12, 10, 8, 25, 12, 
                       26, 25, 18, 11, 10, 20, 20, 24, 20, 
                       9, 17, 27, 10, 18, 8, 9, 24, 20, 17, 
                       21, 23, 21, 24, 27, 10, 12, 14, 27, 13, 
                       13, 16, 27, 26, 9, 10, 23, 14, 8, 25, 16, 22, 21 
                       ]
    C = 8 + 20 + 100
    
    def __init__(self, train=True, HLevel=[1,2,3], image_size=32):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(image_size, image_size),scale=(0.8,1.0)),
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
        self.mat, self.labels = CIFAR100Dataset.load_data(train)
        self.transform = transform_train if train else transform_test

    def __getitem__(self, index):
        image = self.mat[index % len(self.labels)]
        image = np.transpose(image,(1,2,0))
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image).float()
        
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(CIFAR100Dataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = int(self.labels[index])
            c_name = CIFAR100Dataset.classes_finest[c_id]
            ch_id = CIFAR100Dataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = CIFAR100Dataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = CIFAR100Dataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(100, dtype=np.float32)
                target[int(self.labels[index])] = 1
            elif self.HLevel == 2:
                target = np.zeros(20, dtype=np.float32)
                c_id = int(self.labels[index])
                c_name = CIFAR100Dataset.classes_finest[c_id]
                ch_id = CIFAR100Dataset.class_mapping_hm[c_name]
                p_id = CIFAR100Dataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 8] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(8, dtype=np.float32)
                c_id = int(self.labels[index])
                c_name = CIFAR100Dataset.classes_finest[c_id]
                ch_id = CIFAR100Dataset.class_mapping_hm[c_name]
                p_id = CIFAR100Dataset.CLASS_HIERARCHY[ch_id]
                pp_id = CIFAR100Dataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        
        return image, target
    
    def __len__(self):
        return len(self.labels)
    
    def load_data(train=True):
        train_data_path = "/home/wyx/datasets/cifar-100-python/train"
        test_data_path = "/home/wyx/datasets/cifar-100-python/test"
        path = train_data_path if train else test_data_path
        f = open(path, "rb")
        dic = pickle.load(f, encoding="bytes")
        f.close()
        images = dic[b"data"]
        images = np.reshape(images, (-1, 3, 32, 32))
        finelabels = np.asarray(dic[b"fine_labels"])
        return images, finelabels

class CIFAR100DatasetAL(Dataset):
    
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
        path = "/home/wyx/datasets/cifar-100-python/train"
        f = open(path, "rb")
        dic = pickle.load(f, encoding="bytes")
        f.close()
        images = dic[b"data"]
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.transpose(images, (0,2,3,1)).astype(np.uint8)
        finelabels = np.asarray(dic[b"fine_labels"])
        
        C = len(CIFAR100Dataset.classes_finest)
        data = [[] for _ in range(C)] # Rerange by classes
        for i in range(len(finelabels)):
            image, label = images[i], finelabels[i]
            data[int(label)].append((int(label), image))

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
        L = CIFAR100DatasetAL(L_data, "L", HLevel, image_size)
        U = CIFAR100DatasetAL(U_data, "U", HLevel, image_size)
        return L, U

    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,img = self.data[datum_id]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        
        target=None
        if isinstance(self.HLevel, list):
            target = np.zeros(len(CIFAR100Dataset.CLASS_HIERARCHY), dtype=np.float32)
            c_id = label
            c_name = CIFAR100Dataset.classes_finest[c_id]
            ch_id = CIFAR100Dataset.class_mapping_hm[c_name]
            target[ch_id] = 1 # Level-3
            p_id = CIFAR100Dataset.CLASS_HIERARCHY[ch_id]
            target[p_id] = 1 # Level-2
            pp_id = CIFAR100Dataset.CLASS_HIERARCHY[p_id] 
            target[pp_id] = 1 # Level-1
        elif isinstance(self.HLevel, int):
            if self.HLevel == 3:
                target = np.zeros(100, dtype=np.float32)
                target[label] = 1
            elif self.HLevel == 2:
                target = np.zeros(20, dtype=np.float32)
                c_id = label
                c_name = CIFAR100Dataset.classes_finest[c_id]
                ch_id = CIFAR100Dataset.class_mapping_hm[c_name]
                p_id = CIFAR100Dataset.CLASS_HIERARCHY[ch_id]
                target[p_id - 8] = 1 # Modify the offset by ignoring level-1
            elif self.HLevel == 1:
                target = np.zeros(8, dtype=np.float32)
                c_id = label
                c_name = CIFAR100Dataset.classes_finest[c_id]
                ch_id = CIFAR100Dataset.class_mapping_hm[c_name]
                p_id = CIFAR100Dataset.CLASS_HIERARCHY[ch_id]
                pp_id = CIFAR100Dataset.CLASS_HIERARCHY[p_id]
                target[pp_id] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

if __name__ == "__main__":
    ds = CIFAR100Dataset(True,[1,2,3],32)
    image, label = ds[13]
    print(label)
    print(len(label))
    
    