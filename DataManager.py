from PIL import Image, ImageOps
import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms


DS_STORE='.DS_Store'
RGB="RGB"
bg_color="white"
IMG_SIZE = (256,256)

image_database=os.getcwd()+"/Database"
def transformImages():
    msgs=""
    try:
        train_folder=os.getcwd()+"/Database/training-data"
        if os.path.exists(train_folder):
            shutil.rmtree(train_folder)
        for root, folders, files in os.walk(image_database):
            for sub in folders:
                print('processing folder ' + sub)
                newLoc = os.path.join(train_folder,sub)
                if not os.path.exists(newLoc):
                    os.makedirs(newLoc)
                file_names = os.listdir(os.path.join(root,sub))
                for file in file_names:
                    if file!=DS_STORE:
                        img=Image.open(os.path.join(root,sub,file))
                        new_image = Image.new(RGB, IMG_SIZE, bg_color)
                        img.thumbnail(IMG_SIZE,Image.ANTIALIAS)
                        new_image.paste(img, (int((IMG_SIZE[0] - img.size[0]) / 2),int((IMG_SIZE[1]-img.size[1]) / 2)))
                        newPath = os.path.join(newLoc, file)
                        new_image.save(newPath)

        mgs=("Transformation Step complete")
    except:
        mgs=("Error Occured in TransformData")
    return msgs


def loadImages(data_path):
    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transformation)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=50,num_workers=0,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=50,num_workers=0,shuffle=False)
    return train_loader, test_loader
