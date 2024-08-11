import net

from skimage import metrics
from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np
from skimage.color import rgb2ycbcr

def mse1(image1, image2):
    squared_error = torch.pow(image1 - image2, 2)
    return torch.mean(squared_error)

def calculate_psnr(img1, img2):
    img1 = img1.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img2 = img2.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    return metrics.peak_signal_noise_ratio(img1, img2)

def calculate_ssim(img1, img2):
    # Convert torch tensors to numpy arrays and transpose to (H, W, C)
    img1 = img1.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img2 = img2.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    
    # Convert RGB images to YCbCr
    img1_ycbcr = rgb2ycbcr(img1)
    img2_ycbcr = rgb2ycbcr(img2)
    
    # Extract Y channel
    img1_y = img1_ycbcr[..., 0]
    img2_y = img2_ycbcr[..., 0]
    
    # Calculate SSIM on Y channel
    return ssim(img1_y, img2_y, data_range=img1_y.max() - img1_y.min())
################################################################################
import torchvision.transforms as transforms
import torch.nn.functional as F

def SR(device, model, image):
  w, h = image.size
  scale = model.scale

  transform_HR = transforms.Compose([
      transforms.Resize((h,w)),
      transforms.ToTensor(),
  ])

  HR = transform_HR(image)
  HR = HR.to(device)

  transform_LR= transforms.Compose([
      transforms.Resize((h//scale,w//scale)),
      transforms.ToTensor(),
  ])
  LR = transform_LR(image)
  LR = LR.to(device)
  # predict
  with torch.no_grad():
      SR, _ = model(LR.unsqueeze(0))
  # tensor to numpy
  sr_image_np = SR.squeeze().clamp(0, 1).cpu().numpy()

  return sr_image_np

################################################################################

def test_img(device, model, image):
  w_ori, h_ori = image.size

  w_fit = w_ori%64
  h_fit = h_ori%64

#####
  w_crop = w_ori - w_fit
  h_crop = h_ori - h_fit

  image00 = image.crop((0, 0, w_crop, h_crop))
  np_image_00 = SR(device, model, image00) # np [C, H, W]

  final_np = np.zeros((3, h_ori, w_ori))
  final_np[:, :h_crop, :w_crop] = np_image_00

  if w_fit != 0:
    # only w
    image_10 = image.crop((w_fit, 0, w_ori, h_crop))
    np_image_10 = SR(device, model, image_10) # np [C, W, H]
    final_np[:, :h_crop, w_crop:] = np_image_10[:, :, w_crop-w_fit:]

    if(h_fit != 0):
      # both
      image_11 = image.crop((w_fit, h_fit, w_ori, h_ori))
      np_image_11 = SR(device, model, image_11)
      final_np[:, h_crop:, w_crop:] = np_image_11[:, h_crop-h_fit:, w_crop-w_fit:]

      image_01 = image.crop((0, h_fit, w_crop, h_ori))
      np_image_01 = SR(device, model, image_01)
      final_np[:, h_crop:, :w_crop] = np_image_01[:, h_crop-h_fit:, :]

  elif h_fit != 0:
    # only h
    image_01 = image.crop((0, h_fit, w_crop, h_ori))
    np_image_01 = SR(device, model, image_01)
    final_np[:, h_crop:, :w_crop] = np_image_01[:, h_crop-h_fit:, :]

##
  transform_HR = transforms.Compose([
      transforms.ToTensor()
  ])

  HR = transform_HR(image)
##

  image1 = HR
  image2 = torch.from_numpy(final_np)

  # MSE & PSNR & SSIM - with resized HR
  mse_value = mse1(image1, image2)
  psnr_value = calculate_psnr(image1, image2)
  ssim_value = calculate_ssim(image1.squeeze().cpu(), image2.squeeze().cpu())

  HR = HR.squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
  final = final_np.transpose(1, 2, 0)

  return (HR, final),(mse_value.item(), psnr_value.item(), ssim_value)

################################################################################
import os
from PIL import Image

def test_set(folder_path, device, model, save_path=None):
  file_list = os.listdir(folder_path)

  image_files = [file for file in file_list if file.endswith(('.png', '.jpg', '.jpeg'))]

  i = 0
  total_psnr, total_ssim = 0, 0

  for image_file in image_files:
      i += 1
      image_path = os.path.join(folder_path, image_file)

      image = Image.open(image_path)

      (HR, SR), (mse, psnr, ssim) = test_img(device, model, image)
      total_psnr += psnr
      total_ssim += ssim

      if save_path!=None:
        SR = (SR * 255).astype(np.uint8)
        image = Image.fromarray(SR)
        image_file_path = os.path.join(f"{save_path}/{image_file}")
        image.save(image_file_path)


  print(f'psnr: {total_psnr / i}, ssim: {total_ssim / i}')

################################################################################

