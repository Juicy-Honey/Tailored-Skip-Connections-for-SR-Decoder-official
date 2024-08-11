import torch
import os
import zipfile
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# zip_name, bs, scale, im_size=None
def un_zip (name, path="../datasets"):

    zip_path  = f"{path}/{name}.zip"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/dataset')

    hr_path  = f'/content/dataset/{name}' # dataset folder
    return hr_path

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

class Dataset_SR(torch.utils.data.Dataset):
    def __init__(self, hr_filenames, transform=None, transform_LR=None, scale=4):
        self.hr_filenames = hr_filenames
        self.transform    = transform
        self.transform_LR = transform_LR
        self.scale = scale

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, index):
        hr_image = load_image(self.hr_filenames[index])
        lr_image = hr_image

        if self.transform:
            hr_image = self.transform(hr_image)

        if self.transform_LR == None:
            lr_image = F.interpolate(hr_image, scale_factor=1/self.scale)
        else:
            lr_image = self.transform_LR(lr_image)

        return hr_image, lr_image

###########main
def load_ds(zip_name, bs, scale, img_size=None):
  hr_path = un_zip(zip_name)

  if img_size == None:
    transform_HR = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_LR = None
  else:
    transform_HR = transforms.Compose([
      transforms.CenterCrop(img_size),
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor()
    ])
    transform_LR = transforms.Compose([
    transforms.CenterCrop(img_size),
    transforms.Resize((img_size//scale, img_size//scale)),
    transforms.ToTensor(),
    ])
  
  hr_filenames = [os.path.join(hr_path, x) for x in os.listdir(hr_path)]

  dataset_SR = Dataset_SR(hr_filenames, transform_HR, transform_LR)
  print(f'{len(dataset_SR)} images loaded')
  train_loader = torch.utils.data.DataLoader(dataset_SR, batch_size=bs, shuffle=True)

  return train_loader






