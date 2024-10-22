import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

class DAE(nn.Module):
    def __init__(self, img_shape: tuple, latent_dim: int) -> None:
        super(DAE, self).__init__()
        self.img_shape = img_shape
        
        self.linear1 = nn.Linear(np.prod(img_shape), 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.latent1 = nn.Linear(512, latent_dim)
        
        self.latent2 = nn.Linear(latent_dim, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear6 = nn.Linear(128, np.prod(img_shape))
        
        self.relu = nn.LeakyReLU(negative_slope=.1, inplace=True)
        self.tanh = nn.Tanh()
    
    def encode(self, x):
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        return self.relu(self.linear3(h2))
    
    def decoder(self, z):
        h4 = self.relu(self.linear4(z))
        h5 = self.relu(self.linear5(h4))
        return self.tanh(self.linear6(h5))
        
    def forward(self, img):
        og_shape = img.shape
        img_flat = img.view(img.shape[0], -1)
        z = self.encode(img_flat)
        recon = self.decoder(z)
        return recon.view(og_shape)


def pearson_correlation(p1, p2):
    """Pearson correlation method

    Args:
        p1 (np.array): original image
        p2 (np.array): generated image

    Returns:
        float: pearson correlation between images
    """
    return np.corrcoef(p1.flatten(), p2.flatten(), rowvar=False)[0][1]


def stats(arr):
    """Calculates the IQR of array elements.

    Args:
        arr (np.array): array with numeric values

    Returns:
        float: IQR calculated from quartiles
        float: lower bound for anomaly detection
        float: higher bound for anomaly detection
    """
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25
    return iqr, q25 - 1.5 * iqr, q75 + 1.5 * iqr


def novelty_inference(model, data_dir, batch_size, device, img_shape, threshold = 0.963608371833327):
    """Method novelty model inference

    Args:
        model (pytorch model): novelty detection model
        data_dir (str): data path
        batch_size (int): qtd images to process
        device (str): describre the device to process
        img_shape (tuple): image shape
        threshold (float, optional): novelty threshold. Defaults to 0.963608371833327.
    """
    
    out_dir = f"./negative_results"
    os.makedirs(out_dir, exist_ok = True)

    img_transforms = v2.Compose([
        v2.Resize(img_shape[:-1]),
        v2.Grayscale(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    val_set = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=img_transforms)

    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon = model(data)
            break
 
    corr = list(map(pearson_correlation, data.cpu().numpy(), recon.cpu().numpy()))
    
    for i, c in enumerate(corr):
        original_image = data[i].cpu().numpy().transpose((1, 2, 0))

        if c < threshold:
           fig, ax = plt.subplots(1, 1, figsize=(10, 7))
           ax.imshow(original_image, cmap = plt.cm.gray)
           ax.axis('OFF')
           fig.savefig(f"{out_dir}/{i}.jpg", bbox_inches="tight")
           plt.close()
     
            
def pipeline():
    DATA_DIR = '/home/nata-brain/Documents/proj/image-enhancer/datasets/cable_dataset_tester/images/dae_dataset/val/'
    MODEL_OUT = '../models'
    NET_NAME = 'cable_novelty.pt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SHAPE = (300, 600, 1)
    LATENT_DIM = 2
    BATCH_SIZE = 4
    
    model: nn.Module = DAE(IMG_SHAPE, LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_OUT, NET_NAME), 
                                    weights_only=True))
    
    novelty_inference(model = model, data_dir = DATA_DIR, batch_size = BATCH_SIZE, 
                      device = DEVICE, img_shape = IMG_SHAPE, threshold = 0.96361)
    
    
if __name__ == "__main__":
    pipeline()