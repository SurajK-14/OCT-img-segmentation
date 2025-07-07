import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the color map for annotations
color_map = {
    (255, 0, 0, 255): 1,    # ILM: Red
    (0, 0, 252, 255): 2     # RPE: Blue (252)
}

def calculate_vertical_distance(annotation_image):
    # Convert the image to RGBA format
    annotation_image = annotation_image.convert("RGBA")
    annotation_array = np.array(annotation_image)

    # Initialize arrays to store the y-coordinates of ILM and RPE layers
    ilm_y_coords = np.zeros(annotation_array.shape[1])
    rpe_y_coords = np.zeros(annotation_array.shape[1])

    # Iterate over each column to find the y-coordinates of ILM and RPE layers
    for x in range(annotation_array.shape[1]):
        for y in range(annotation_array.shape[0]):
            pixel = tuple(annotation_array[y, x])
            if pixel == (255, 0, 0, 255):  # ILM layer
                ilm_y_coords[x] = y
            elif pixel == (0, 0, 252, 255):  # RPE layer
                rpe_y_coords[x] = y

    # Calculate the vertical distance between ILM and RPE layers for each column
    vertical_distances = np.abs(ilm_y_coords - rpe_y_coords)
    
    return vertical_distances

def generate_depth_map(raw_images_folder, annotated_images_folder):
    depth_map = []

    # Iterate over each image in the annotated images folder
    for filename in os.listdir(annotated_images_folder):
        if filename.endswith(".png"):
            #raw_image_path = os.path.join(raw_images_folder, filename)
            annotated_image_path = os.path.join(annotated_images_folder, filename)

            # Open the raw and annotated images
            #raw_image = Image.open(raw_image_path)
            annotated_image = Image.open(annotated_image_path)

            # Calculate the vertical distances for the annotated image
            vertical_distances = calculate_vertical_distance(annotated_image)

            # Append the vertical distances to the depth map
            depth_map.append(vertical_distances)

    return depth_map

def plot_depth_map(depth_map):
    plt.figure(figsize=(10, 6))

    for distances in depth_map:
        plt.scatter(range(len(distances)), [len(distances)] * len(distances), c=distances, cmap='RdYlGn')

    plt.colorbar(label='Vertical Distance (pixels)')
    plt.xlabel('Image Length (pixels)')
    plt.ylabel('Number of Images')
    plt.title('3D Depth Map of Retinal OCT Images')
    plt.show()

# Define the folders containing raw images and annotated images
image_dir = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD_PNG/image"
annotated_images_folder = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD_PNG/layer"




# Generate the depth map
depth_map = generate_depth_map(image_dir, annotated_images_folder)

# Plot the depth map
plot_depth_map(depth_map)