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

    # Create boolean masks for each layer color np.all(..., axis=2) checks for a color match across the RGBA channels
    ilm_mask = np.all(annotation_array == ILM_COLOR, axis=2)
    rpe_mask = np.all(annotation_array == RPE_COLOR, axis=2)

    # Find the y-coordinate (row index) of the first occurrence of each layer in every column (axis=0).
    ilm_y_coords = np.argmax(ilm_mask, axis=0)
    rpe_y_coords = np.argmax(rpe_mask, axis=0)

    # np.argmax returns 0 if the layer isn't found in a column. We must replace these with NaN to indicate missing data.
    # We create a mask for columns where the layers were actually found.
    ilm_found = np.any(ilm_mask, axis=0)
    rpe_found = np.any(rpe_mask, axis=0)

    # Use np.where to keep the coordinate if found, otherwise set to NaN
    ilm_y_coords = np.where(ilm_found, ilm_y_coords, np.nan)
    rpe_y_coords = np.where(rpe_found, rpe_y_coords, np.nan)

    # Calculate the absolute vertical distance. The result will be NaN for any column where at least one layer was not found.
    vertical_distances = np.abs(rpe_y_coords - ilm_y_coords)

    return vertical_distances

def generate_thickness_map(annotated_images_folder: str, patient_id: str = None) -> np.ndarray:
    """
    Generates a 2D thickness map from a folder of annotated B-scans.

    Args:
        annotated_images_folder: Path to the folder with annotated images.
        patient_id: Optional. The ID of the patient to process (e.g., "AMD_001").
                    If None, all images in the folder are processed.

    Returns:
        A 2D NumPy array where each row corresponds to a B-scan and each
        column corresponds to the retinal thickness at that location.
    """
    # Get a sorted list of all .png files to ensure correct processing order
    all_image_files = sorted([f for f in os.listdir(annotated_images_folder) if f.endswith(".png")])

    if patient_id:
        # Filter files for the specified patient. The pattern is "AMD_001_024.png".
        image_files = [f for f in all_image_files if f.startswith(f"{patient_id}_")]
        tqdm_desc = f"Patient {patient_id}"
    else:
        image_files = all_image_files
        tqdm_desc = "Calculating Distances"

    if not image_files:
        if patient_id:
            print(f"Warning: No .png files found for patient '{patient_id}' in {annotated_images_folder}")
        else:
            print(f"Warning: No .png files found in {annotated_images_folder}")
        return np.array([])

    all_distances = []
    print(f"Processing {len(image_files)} annotated images...")
    # Use tqdm for a progress bar
    for filename in tqdm(image_files, desc=tqdm_desc):
        annotated_image_path = os.path.join(annotated_images_folder, filename)
        try:
            with Image.open(annotated_image_path) as annotated_image:
                vertical_distances = calculate_vertical_distance_vectorized(annotated_image)
                all_distances.append(vertical_distances)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return np.array(all_distances)

def plot_thickness_map(thickness_map: np.ndarray, title: str = "Retinal Thickness Map", output_path: str = None) -> None:
    """
    Plots the 2D retinal thickness map.

    Args:
        thickness_map: The 2D NumPy array of thickness values.
        /home/suraj/Git/OCT-img-segmentation/output
    """
    if thickness_map.size == 0:
        print("Cannot plot an empty thickness map.")
        return

    plt.figure(figsize=(12, 8))

    im = plt.imshow(thickness_map, cmap='RdYlGn', aspect='auto', interpolation='none')

    plt.colorbar(im, label='Retinal Thickness (pixels)')
    plt.xlabel('A-Scan Location (Pixel Column)')
    plt.ylabel('B-Scan Index (Image Number)')
    #plt.title('Retinal Thickness Map from OCT B-Scans')
    #plt.show()
    plt.title('Retinal Thickness Map from OCT B-Scans')
    
    if output_path:
        output_file = os.path.join(output_path, 'thickness_map.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Thickness map saved to {output_file}")

def main():
    """Main function to run the script."""
    # --- Configuration ---
    # Annotated image path
    annotated_images_folder = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD_PNG/layer"
    
    #output map path
    output_folder = '/home/suraj/Git/OCT-img-segmentation/output'
    # Set patient_id to a specific ID (e.g., "AMD_001") to process only that patient's scans.
    # Set to None to process all scans in the folder.
    patient_id_to_process = "AMD_001"

    # Generate the thickness map
    thickness_map = generate_thickness_map(annotated_images_folder, patient_id=patient_id_to_process)

    # Plot the thickness map
    if thickness_map.any():
        #plot_thickness_map(thickness_map)
        if patient_id_to_process:
            plot_title = f"Thickness Map for Patient {patient_id_to_process}"
            output_filename = f"{patient_id_to_process}_thickness_map.png"
        else:
            plot_title = "Thickness Map for All Patients"
            output_filename = "thickness_map_all_patients.png"
        
        print(f"Thickness map generated for patient '{patient_id_to_process}'.")
        
        plot_thickness_map(thickness_map, title=plot_title, output_path=output_folder)
        

if __name__ == "__main__":
    main()
