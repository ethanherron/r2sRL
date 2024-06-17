import fire
import torch, os, random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from einops import repeat
import cv2
from PIL import Image
from unet import UNetModel

def pad(tensor):
    return repeat(tensor, 'b -> b 1 1 1')



class CIFAR10CannyEdgeDataset(Dataset):
    def __init__(self, root, train=True, download=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        
        transform_list = [
            transforms.Resize((120, 160)),  # Resizing to a larger size for demonstration purposes
            transforms.ToTensor()
        ]
        
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.convert('RGB')  # Ensure the image is in RGB mode
        image_tensor = self.transform(image)

        # Convert image to grayscale and to numpy array for Canny edge detection
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray_image = cv2.resize(gray_image, (160, 120))  # Resizing to match the resized RGB image

        edges = cv2.Canny(gray_image, 100, 200)
        edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0

        return image_tensor, edges_tensor
    
    
class CIFAR10CannyEdgeDoubleImageDataset(Dataset):
    def __init__(self, root, train=True, download=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        
        transform_list = [
            transforms.Resize((120, 160)),  # Resizing to a larger size for demonstration purposes
            transforms.ToTensor()
        ]
        
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the first image
        x0, label1 = self.dataset[idx]
        x0 = x0.convert('RGB')  # Ensure the image is in RGB mode
        x0_tensor = self.transform(x0)

        # Randomly select another image
        idx2 = random.randint(0, len(self.dataset) - 1)
        x1, label2 = self.dataset[idx2]
        x1 = x1.convert('RGB')
        x1_tensor = self.transform(x1)

        # Convert the second image to grayscale and to numpy array for Canny edge detection
        gray_x1 = cv2.cvtColor(np.array(x1), cv2.COLOR_RGB2GRAY)
        gray_x1 = cv2.resize(gray_x1, (160, 120))  # Resizing to match the resized RGB image

        edges2 = cv2.Canny(gray_x1, 100, 200)
        edges2_tensor = torch.tensor(edges2, dtype=torch.float32).unsqueeze(0) / 255.0

        return x0_tensor, x1_tensor, edges2_tensor
    

class UnpairedCIFAR10SimImages(Dataset):
    def __init__(self, cifar10_root, sim_images_dir):
        self.cifar10_dataset = datasets.CIFAR10(root=cifar10_root, train=True, download=True)
        self.sim_images_dir = sim_images_dir
        self.sim_images_files = [f for f in os.listdir(sim_images_dir) if f.endswith('.png')]
        
        transform_list = [
            transforms.Resize((120, 160)),  # Resize to 160x120
            transforms.ToTensor()
        ]
        
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.sim_images_files)

    def __getitem__(self, idx):
        # Randomly select an image from CIFAR-10
        cifar_idx = random.randint(0, len(self.cifar10_dataset) - 1)
        x0, _ = self.cifar10_dataset[cifar_idx]
        x0 = x0.convert('RGB')  # Ensure the image is in RGB mode
        x0_tensor = self.transform(x0)

        # Get the corresponding sim image
        sim_image_path = os.path.join(self.sim_images_dir, self.sim_images_files[idx])
        x1 = Image.open(sim_image_path).convert('RGB')
        x1_tensor = self.transform(x1)

        # Convert the sim image to grayscale and to numpy array for Canny edge detection
        gray_x1 = cv2.cvtColor(np.array(x1), cv2.COLOR_RGB2GRAY)
        gray_x1 = cv2.resize(gray_x1, (160, 120))  # Resize to match the resized RGB image

        x1_canny = cv2.Canny(gray_x1, 100, 200)
        x1_canny_tensor = torch.tensor(x1_canny, dtype=torch.float32).unsqueeze(0) / 255.0

        return x0_tensor, x1_tensor, x1_canny_tensor
    


class RectifiedFlow():
    def __init__(self, model=None, device=None, num_steps=10):
        self.device = device
        self.model = model.to(self.device)
        self.N = num_steps
        
    def get_train_tuple(self, x0=None, x1=None):
        # randomly sample timesteps for training - timesteps are analogous to 
        # points along the linear interpolation of x0 and x1.
        t = torch.rand((x0.shape[0])).to(self.device)
        t = F.sigmoid(t)
        
        # find interpolated x i.e., x_t
        x_t = pad(t) * x1 + (1. - pad(t)) * x0
        
        # find our ground truth target value (velocity) we want our network to
        # approximate. This velocity term is the time derivative of the linear 
        # interpolation above. ie dX_t/dt = d(t*x1 + (1-t)*x0)/dt
        velocity = x1 - x0
        
        return x_t, t, velocity
    
    def rectified_flow_loss(self, x0, x1, condition):
        '''
        Loss function for rectified flow model.

        x0: input tensor of shape (batch_size, channels, height, width) Real Images
        x1: input tensor of shape (batch_size, channels, height, width) Sim Images
        condition: input tensor of shape (batch_size, channels, height, width) Canny Edge Images
        
        output: loss value we will optimize params of self.model with.
        '''
        # initialize x0 and x1 and send to device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        condition = condition.to(self.device)
        
        # get inputs (x_t and t) for network and velocity value for loss function.
        xt, t, velocity = self.get_train_tuple(x0, x1)
        
        # make velocity prediction with network
        input = torch.cat([xt, condition], dim=1)
        velocity_hat = self.model(input, t)
        
        # compute loss between prediction and velocity and return
        return F.mse_loss(velocity_hat, velocity)
        
    @torch.no_grad()
    def sample_ode(self, x0=None, condition=None, N=None):
        # initialize number of timesteps in ode solver
        if N is None:
            N = self.N
            
        # initialize delta t
        dt = 1./N
        
        # initialize x for solver
        x = x0.detach().clone().to(self.device)
        condition = condition.to(self.device)
        
        # Euler method integration scheme
        for i in range(N):
            # init timesteps and send to device
            t = torch.ones((x0.shape[0])) * i / N
            t = t.to(self.device)
            
            #make velocity prediction
            input = torch.cat([x, condition], dim=1)
            velocity = self.model(input, t)
            
            #update x_t+1
            x = x.detach().clone() + velocity * dt
            
        return x
    

