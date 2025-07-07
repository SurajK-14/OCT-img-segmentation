
# %% Import statements
import os
import re
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from transformers import AutoImageProcessor, TrainingArguments, Trainer, RTDetrV2ForObjectDetection, RTDetrImageProcessor
import albumentations as A

# %% Define paths

BASE_DIR = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD_PNG"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")
EMBEDDING_SAVE_DIR = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD/output/embeddings"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Adjust based on GPU memory

# %% Functions

def create_smaller_dataset(image_dir, mask_dir, target_size=5000, n_remove=15, 
                           fovea_range=(40, 60), sample_step=3)-> tuple:
    ''' 
    Creates smaller dataset from the original dataset.
    Excludes first and last n_remove images, selects ~18-20 images per patient,
    prioritizes fovea-centered images, and samples remaining images to avoid redundancy.
    Args:
        image_dir (str): Directory containing images.
        mask_dir (str): Directory containing masks.
        target_size (int): Target number of images.
        n_remove (int): Number of images to remove from the start and end.
        fovea_range (tuple): Range of indices for fovea-centered images.
        sample_step (int): Step size for sampling remaining images.
    Returns:
        tuple: Lists of selected image paths
        tuple: Lists of corresponding mask paths.
    '''
    # Step 1: Collect all valid files without deleting
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    patient_dict = {}
    pattern = re.compile(r'(AMD_\d{3})_(\d{3})\.png')
    for f in files:
        m = pattern.match(f)
        if m:
            patient, idx = m.group(1), int(m.group(2))
            if n_remove < idx <= (100 - n_remove):  # Exclude first and last n_remove
                patient_dict.setdefault(patient, []).append((idx, f))
    
    # Step 2: Select ~18-20 images per patient
    selected_images = []
    selected_masks = []
    images_per_patient = max(1, target_size // len(patient_dict))  # ~18-20 images per patient
    
    
    
def get_image_paths(image_dir, max_images=300):
    """
    Collects up to `max_images` image file paths from the specified directory.

    Args:
        image_dir (str): Directory containing image files.
        max_images (int): Maximum number of images to collect.

    Returns:
        List[str]: List of image file paths.
    """
    image_paths = sorted([str(p) for p in Path(image_dir).glob("*.png")])
    return image_paths[:max_images]

def load_and_preprocess_images(image_paths, processor):
    """
    Loads images, converts grayscale to 3-channel RGB, and preprocesses them for the model.

    Args:
        image_paths (List[str]): List of image file paths.
        processor: Hugging Face image processor.

    Returns:
        torch.Tensor: Batch of preprocessed images.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")  # Ensures 3 channels
        images.append(img)
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"]

def save_embeddings(embeddings, image_paths, save_dir):
    """
    Saves embeddings as .npz files, one per image.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        image_paths (List[str]): Corresponding image file paths.
        save_dir (str): Directory to save .npz files.
    """
    os.makedirs(save_dir, exist_ok=True)
    for emb, path in zip(embeddings, image_paths):
        fname = Path(path).stem + ".npz"
        np.savez_compressed(os.path.join(save_dir, fname), embedding=emb)

# =========================
# Main Extraction Function
# =========================

def extract_embeddings(image_dir, save_dir, model_name, batch_size=16, max_images=300):
    """
    Extracts embeddings from images using a pretrained ViT model and saves them.

    Args:
        image_dir (str): Directory with input images.
        save_dir (str): Directory to save embeddings.
        model_name (str): Hugging Face model name.
        batch_size (int): Batch size for processing.
        max_images (int): Number of images to process.
    """
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    # Get image paths
    image_paths = get_image_paths(image_dir, max_images)
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        batch_pixels = load_and_preprocess_images(batch_paths, processor).to(DEVICE)
        with torch.no_grad():
            outputs = model(batch_pixels)
            # Use [CLS] token as embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        save_embeddings(embeddings, batch_paths, save_dir)

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    """
    Main script entry point. Extracts embeddings from a subset of images and saves them.
    """
    extract_embeddings(
        image_dir=IMAGE_DIR,
        save_dir=EMBEDDING_SAVE_DIR,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        max_images=300
    )