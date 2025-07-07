import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# --- Constants ---
# Define the colors for the annotated layers in RGBA format.
# Tuple[int, int, int, int]
ILM_COLOR: Tuple[int, int, int, int] = (255, 0, 0, 255)  # Inner Limiting Membrane: Red
RPE_COLOR: Tuple[int, int, int, int] = (0, 0, 252, 255)  # Retinal Pigment Epithelium: Blue

def calculate_vertical_distance_vectorized(annotation_image: Image.Image) -> np.ndarray:
    """
    Calculates the vertical pixel distance between two annotated layers in an OCT
    B-scan using a fast, vectorized NumPy approach.

    Args:
        annotation_image: A PIL Image object of the annotated B-scan.

    Returns:
        A 1D NumPy array containing the vertical distance for each column.
        If a layer is not found in a column, the distance will be NaN.
    """
    # Convert image to a NumPy array for efficient processing
    annotation_array = np.array(annotation_image.convert("RGBA"))

    # Create boolean masks for each layer color
    # np.all(..., axis=2) checks for a color match across the RGBA channels
    ilm_mask = np.all(annotation_array == ILM_COLOR, axis=2)
    rpe_mask = np.all(annotation_array == RPE_COLOR, axis=2)

    # Find the y-coordinate (row index) of the first occurrence of each layer
    # in every column (axis=0).
    ilm_y_coords = np.argmax(ilm_mask, axis=0)
    rpe_y_coords = np.argmax(rpe_mask, axis=0)

    # np.argmax returns 0 if the layer isn't found in a column. We must replace
    # these with NaN to indicate missing data.
    # We create a mask for columns where the layers were actually found.
    ilm_found = np.any(ilm_mask, axis=0)
    rpe_found = np.any(rpe_mask, axis=0)

    # Use np.where to keep the coordinate if found, otherwise set to NaN
    ilm_y_coords = np.where(ilm_found, ilm_y_coords, np.nan)
    rpe_y_coords = np.where(rpe_found, rpe_y_coords, np.nan)

    # Calculate the absolute vertical distance. The result will be NaN for any
    # column where at least one layer was not found.
    vertical_distances = np.abs(rpe_y_coords - ilm_y_coords)

    return vertical_distances

def generate_thickness_map(annotated_images_folder: str) -> np.ndarray:
    """
    Generates a 2D thickness map from a folder of annotated B-scans.

    Args:
        annotated_images_folder: Path to the folder with annotated images.

    Returns:
        A 2D NumPy array where each row corresponds to a B-scan and each
        column corresponds to the retinal thickness at that location.
    """
    # Get a sorted list of image files to ensure correct processing order
    # (e.g., scan_1.png, scan_2.png, ..., scan_10.png)
    image_files = sorted([f for f in os.listdir(annotated_images_folder) if f.endswith(".png")])

    if not image_files:
        print(f"Warning: No .png files found in {annotated_images_folder}")
        return np.array([])

    all_distances = []
    print(f"Processing {len(image_files)} annotated images...")
    # Use tqdm for a progress bar
    for filename in tqdm(image_files, desc="Calculating Distances"):
        annotated_image_path = os.path.join(annotated_images_folder, filename)
        try:
            with Image.open(annotated_image_path) as annotated_image:
                vertical_distances = calculate_vertical_distance_vectorized(annotated_image)
                all_distances.append(vertical_distances)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return np.array(all_distances)

def plot_thickness_map(thickness_map: np.ndarray):
    """
    Plots the 2D retinal thickness map.

    Args:
        thickness_map: The 2D NumPy array of thickness values.
    """
    if thickness_map.size == 0:
        print("Cannot plot an empty thickness map.")
        return

    plt.figure(figsize=(12, 8))
    # Use imshow for efficient and correct display of 2D data.
    # 'RdYlGn' colormap: Red (low values/thin) -> Yellow -> Green (high values/thick)
    # 'aspect=auto' allows the plot to fit the figure dimensions.
    # 'interpolation=none' shows raw pixels without blurring.
    im = plt.imshow(thickness_map, cmap='RdYlGn', aspect='auto', interpolation='none')

    plt.colorbar(im, label='Retinal Thickness (pixels)')
    plt.xlabel('A-Scan Location (Pixel Column)')
    plt.ylabel('B-Scan Index (Image Number)')
    plt.title('Retinal Thickness Map from OCT B-Scans')
    plt.show()

def main():
    """Main function to run the script."""
    # --- Configuration ---
    # Please update these paths to match your folder structure
    annotated_images_folder = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD_PNG/layer"

    # Generate the thickness map
    thickness_map = generate_thickness_map(annotated_images_folder)

    # Plot the thickness map
    if thickness_map.any():
        plot_thickness_map(thickness_map)

if __name__ == "__main__":
    main()

