
# %% Imports
import os
import scipy.io as sio
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
import torch

# %% 1. Data Loading and Setup
data_dir = '/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD'
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')]
output_dir = os.path.join(data_dir, 'embeddings')
os.makedirs(output_dir, exist_ok=True)

# %% Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %% Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
model.eval()

# %% READ MULTIPLE .MAT FILES one after another
def read_mat_file(file_path)-> dict:
    try:
        data = sio.loadmat(file_path, squeeze_me=True)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# %% Process each .mat file
for file_path in data_files:
    print(f"Processing file: {file_path}")
    data = read_mat_file(file_path)
    if data is None:
        continue

    images = data.get('images') # taking the 'images' key from the .mat file 

    print("Images shape:", images.shape) #Shape: (512, 1000, 100); 512 is the height, 1000 is the width, and 100 is the number of b-scans

    # Pre-allocate and process b-scans
    b_scans = torch.zeros((images.shape[2], 3, 224, 224), dtype=torch.float32, device=device)
    for i in range(images.shape[2]):
        b_scan = images[:, :, i]
        b_scan_rgb = np.stack([b_scan] * 3, axis=-1) # Convert to RGB
        b_scan_rgb = (b_scan_rgb / b_scan_rgb.max() * 255).astype(np.uint8) # Normalize and convert to uint8
        img = Image.fromarray(b_scan_rgb) # Convert to PIL Image
        b_scans[i] = transform(img).to(device) # Apply transformations

    #Load Model and Extract Embeddings
    with torch.no_grad():
        outputs = model(pixel_values=b_scans)  # Shape: (100, 3, 224, 224) as batch
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    # Save embeddings
    file_name = os.path.basename(file_path).replace('.mat', '_embeddings.npy')
    output_path = os.path.join(output_dir, file_name)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")
# %%
