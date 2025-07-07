# Code to extract vector embeddings from the current OCT image dataset


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
    
    
def extract_embeddings():
    '''
    Extracts vector embeddings from the current OCT image dataset.
    This function should implement the logic to extract embeddings from images.
    It is a placeholder for now and should be implemented based on the specific requirements.
    '''
    pass  # Placeholder for embedding extraction logic
    
def 
    