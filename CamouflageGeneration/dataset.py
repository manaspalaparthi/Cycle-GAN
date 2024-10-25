import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self,root_zebra,root_horse,transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        self.length_dataset = max(len(self.root_horse),len(self.zebra_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self,index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra,zebra_img)
        horse_path = os.path.join(self.root_horse,horse_img)
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image = zebra_img, image0 = horse_img)
            zebra_img = augmentations['image']
            horse_img = augmentations['image0']
        return zebra_img,horse_img

class EnvironmentDataset(Dataset):
    def __init__(self,places,textures,transform=None):
        self.places = places
        self.textures = textures
        self.transform = transform

        self.places_images = os.listdir(places)
        self.textures_images = os.listdir(textures)
        self.length_dataset = max(len(self.places_images),len(self.textures_images))
        self.places_len = len(self.places_images)
        self.textures_len = len(self.textures_images)
    def __len__(self):
        return self.length_dataset

    def __getitem__(self,index):
        places_img = self.places_images[index % self.places_len]
        textures_img = self.textures_images[index % self.textures_len]

        places_path = os.path.join(self.places,places_img)
        textures_path = os.path.join(self.textures,textures_img)
        places_img = np.array(Image.open(places_path).convert("RGB"))
        textures_img = np.array(Image.open(textures_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image = places_img, image0 = textures_img)
            places_img = augmentations['image']
            textures_img = augmentations['image0']
        return places_img,textures_img



if __name__ == "__main__":

    transform = None

    dataset = EnvironmentDataset("dataset/places/","dataset/texture_gray/",transform=transform)

    places_img,textures_img = dataset[0]

    print(places_img.shape)
    print(textures_img.shape)