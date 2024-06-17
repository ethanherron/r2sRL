import fire
import torch, os, random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from einops import repeat
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from unet import UNetModel
import wandb
from rf import RectifiedFlow, CIFAR10CannyEdgeDataset, CIFAR10CannyEdgeDoubleImageDataset, UnpairedCIFAR10SimImages


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

    
def save_models(models, epoch):
    for model in models:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{model.__class__.__name__}_{epoch}.pt')) 
    
def pad(tensor):
    return repeat(tensor, 'b -> b 1 1 1')


def train_rectified_flow(data_loader, rectified_flow, opt):
    rectified_flow.model.train()
    running_loss = 0.0

    # Use tqdm to create a progress bar
    for data in tqdm(data_loader, desc="Training", unit="batch"):
        x0, x1, canny_edge = data
        loss = rectified_flow.rectified_flow_loss(x0, x1, canny_edge)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        wandb.log({'rf_loss_cifar10_2_sim': loss.item()})
        running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)
    return avg_loss

def eval_rectified_flow(data_loader, rectified_flow):
    rectified_flow.model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            x0, x1, canny_edge = data
            pred_x1 = rectified_flow.sample_ode(x0, canny_edge)

            # Only process the first batch
            break

    # Convert tensors to numpy arrays for visualization
    x1_np = x1.permute(0, 2, 3, 1).cpu().numpy()
    pred_x1_np = pred_x1.permute(0, 2, 3, 1).cpu().numpy()

    # Plot the images
    fig, axes = plt.subplots(nrows=2, ncols=x1.size(0), figsize=(15*5, 5*5))
    for i in range(x1.size(0)):
        axes[0, i].imshow(x1_np[i])
        axes[0, i].set_title(f"Ground Truth x1 - {i}")
        axes[0, i].axis('off')

        axes[1, i].imshow(pred_x1_np[i])
        axes[1, i].set_title(f"Predicted x1 - {i}")
        axes[1, i].axis('off')

    # Log images to wandb
    wandb.log({
        "eval_images_cifar10_2_sim": [wandb.Image(fig)]
    })
    plt.close(fig)



wandb.init(
    project="real2sim"
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

cifar_root = '/data/cifar10'
sim_root = '/data/real_to_sim_data/unpaired_sim_images/imagedata'

## CIFAR10 Canny Edge Dataset
# train_dataset = CIFAR10CannyEdgeDataset(root=cifar_root, train=True, download=False)
# val_dataset = CIFAR10CannyEdgeDataset(root=cifar_root, train=False, download=False)
# train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


## CIFAR10 Canny Edge Double Image Dataset
# train_dataset = CIFAR10CannyEdgeDoubleImageDataset(root=cifar_root, train=True, download=False)
# val_dataset = CIFAR10CannyEdgeDoubleImageDataset(root=cifar_root, train=False, download=False)
# train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_data_loader = DataLoader(val_dataset, batch_size=9, shuffle=False)


## Unpaired CIFAR10 and Sim Images
train_dataset = UnpairedCIFAR10SimImages(cifar_root, sim_root, train=True)
train_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_dataset = UnpairedCIFAR10SimImages(cifar_root, sim_root, train=False)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = UNetModel(in_channels=3+1, out_channels=3)

# load cifar10 pretrained model
# model.load_state_dict(torch.load('/data/edherron/r2sRL/wandb/run-20240604_192546-3o3bxdce/files/UNetModel_8.pt'))
# model.load_state_dict(torch.load('/data/edherron/r2sRL/wandb/run-20240604_211806-y1gb9ssm/files/UNetModel_9.pt'))

rf = RectifiedFlow(model=model, device=device, num_steps=10)

print('number of parameters:', sum(p.numel() for p in model.parameters()))

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(10):
    loss_rec = train_rectified_flow(train_data_loader, rf, opt)
    eval_rectified_flow(val_data_loader, rf)
    print('average loss from epoch', i, ' : ', loss_rec)
    
    save_models([model], i)
    
wandb.finish()
