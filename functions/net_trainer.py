import net_tester
import data_loader

import torch.nn.functional as F
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import math

def trainer(device, model, optimizer, num_epochs, dataloader, Model_path, Model_name):

  st_epoch = 0
  set14_path = data_loader.un_zip("set14")
  bsd_path   = data_loader.un_zip("BSD100")

  criterion = nn.MSELoss()
  # criterion = nn.L1Loss()

  for epoch in range(num_epochs):
      total_loss = 0
      total_mse = 0

      batch_count = 0
      num_steps = int(len(dataloader))

      tqdm_dataloader = tqdm(dataloader, desc=f'Epoch {epoch+1+st_epoch}/{num_epochs}', leave=False)

      step_loss = 0
      step_mse = 0

      for step, (hr_images, lr_images) in enumerate(tqdm_dataloader):
          if step >= num_steps:
              break

          st = 500
          if (step % st == 0) and (step != 0):
            # Save Model
            torch.save(model.state_dict(), os.path.join(Model_path, f'{Model_name}_e{epoch+1+st_epoch}_s{step//st}.pth'))
            # Validate
            model.eval()
            print(" ")
            print(" ")
            print(f"[step {step//st}*{st}] set14 & bsd")
            print(f'[step {step}], Loss: {step_loss/st:.4f}, MSE: {step_mse/st:.4f}')
            net_tester.test_set(set14_path, device, model, save_path=f"../outputs/temp/{Model_name}")
            net_tester.test_set(bsd_path  , device, model, save_path=f"../outputs/temp/{Model_name}")
            step_loss = 0
            step_mse = 0
            print("\n\n")

          model.train()

          batch_count += 1

########### GT , LR
          hr_images = hr_images.to(device)
          lr_images = lr_images.to(device)
  ######### predict!
          sr_images, out = model(lr_images) # Y Channel img
          
          # channel counts
          a_mse = 85      # 1
          a = [4, 2, 1, 0.5, 0.5] # 64 128 256 512 512

  ######### losses
          # mse
          mse = criterion(sr_images, hr_images) 
          loss = mse * a_mse

          enc = []
          enc.append(model.vgg_1)
          enc.append(model.vgg_2)
          enc.append(model.vgg_3)
          if model.depth >= 4:
            enc.append(model.vgg_4)
          if model.depth >= 5:
            enc.append(model.vgg_5)

          # loss: decoder
          for i in range(model.depth):                   
            l = criterion(out[i], enc[i](hr_images)) * a[i]
            loss += l

          # loss: SR
          for i in range(int(math.log(model.scale, 2))): 
            idx = i + model.depth
            hr_resized = F.interpolate(hr_images, scale_factor=1/(2**(i+1)), mode='bicubic', align_corners=False).to(device)
            l = criterion(out[idx], enc[model.depth-1](hr_resized))   * a[model.depth-1]
            loss += l

  ######### step
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

  ######### Update Status
          total_loss += loss.item()
          total_mse  += mse.item()

          tqdm_dataloader.set_postfix({'Loss': total_loss/batch_count, 'MSE': total_mse/batch_count })
      print(f'Epoch [{epoch+1+st_epoch}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, MSE: {total_mse/len(dataloader):.4f}')

      # Save Model
      torch.save(model.state_dict(), os.path.join(Model_path, f'{Model_name}_e{epoch+1+st_epoch}.pth'))

      # Validate
      model.eval()
      print("set14")
      net_tester.test_set(set14_path, device, model, save_path=f"../outputs/temp/{Model_name}")
      print("bsd")
      net_tester.test_set(bsd_path  , device, model, save_path=f"../outputs/temp/{Model_name}")
