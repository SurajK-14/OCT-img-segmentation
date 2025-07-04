#!/usr/bin/env python

# %% IMPORTS
import os
import re
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, Image as HFImage
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from transformers import TrainingArguments, Trainer, SegformerForSemanticSegmentation, RTDetrImageProcessor
# import albumentations as A
# from evaluate import load as load_metric

# %% SESSION VARIABLES
# Define paths
base_dir    = "/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD/output"
image_dir   = os.path.join(base_dir, "image")
mask_dir    = os.path.join(base_dir, "layer")
output_dir  = "./segformer-oct"

# %% FUNCTIONS 
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
    
    for patient, idx_files in patient_dict.items():
        idx_files.sort()
        patient_images = []
        patient_masks = []
        # Prioritize fovea-centered images (40-60)
        fovea_images = [(idx, f) for idx, f in idx_files if fovea_range[0] <= idx <= fovea_range[1]]
        # Sample remaining images to avoid redundancy
        other_images = [(idx, f) for idx, f in idx_files if idx < fovea_range[0] or idx > fovea_range[1]]
        sampled_images = other_images[::sample_step]
        
        # Combine and limit to images_per_patient
        selected = fovea_images[:images_per_patient]
        selected += sampled_images[:(images_per_patient - len(selected))]
        
        for idx, fname in selected:
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                selected_images.append(img_path)
                selected_masks.append(mask_path)
    
    print(f"Selected {len(selected_images)} images and {len(selected_masks)} masks")
    return selected_images, selected_masks

def create_dataset(image_paths, mask_paths)-> Dataset:
    ''' 
    Create Dataset objects
    Args:
        image_paths (list): List of image file paths.
        mask_paths (list): List of corresponding mask file paths.
    Returns:
        Dataset: A Hugging Face Dataset object containing images and masks.
    '''
    x = Dataset.from_dict({"image": image_paths, "label": mask_paths})
    x = x.cast_column("image", HFImage())
    x = x.cast_column("label", HFImage())
    return x

def preprocess_mask(mask, target_size=(512, 1000)):
    ''' 
    Preprocess annotations to create integer-valued masks
    '''
    mask = np.array(mask)
    # Resize mask to match image size
    mask_img = Image.fromarray(mask).resize(target_size[::-1], resample=Image.NEAREST)
    mask = np.array(mask_img)
    
    if mask.ndim == 3 and mask.shape[-1] == 4:  # Handle RGBA
        output_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, label in color_map.items():
            # Compare only RGB channels, ensure alpha=255
            matches = np.all(mask[..., :3] == color[:3], axis=-1) & (mask[..., 3] == 255)
            output_mask[matches] = label
        return output_mask
    elif mask.ndim == 3 and mask.shape[-1] == 3:  # Handle RGB
        output_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, label in color_map.items():
            matches = np.all(mask == color[:3], axis=-1)
            output_mask[matches] = label
        return output_mask
    return mask  # If already integer-valued

def preprocess_data(examples):
    image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    target_size = (512, 512)
    images, masks = [], []

    for i, (image, mask) in enumerate(zip(examples["image"], examples["label"])):
        try:
            # Always convert to RGB before resizing
            img_pil = Image.fromarray(np.array(image)).convert("RGB").resize(target_size[::-1], resample=Image.BILINEAR)
            img = np.array(img_pil)
            msk = np.array(Image.fromarray(np.array(mask)).resize(target_size[::-1], resample=Image.NEAREST))
            processed_mask = preprocess_mask(msk, target_size=target_size)
            images.append(img)
            masks.append(processed_mask)
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            images.append(np.zeros((*target_size, 3), dtype=np.uint8))
            masks.append(np.zeros(target_size, dtype=np.uint8))

    encoding = image_processor(images, return_tensors="pt", do_normalize=True)
    #encoding["labels"] = torch.tensor(masks, dtype=torch.long)
    encoding["labels"] = torch.tensor(np.array(masks), dtype=torch.long)
    return encoding

def compute_metrics(eval_pred):
    '''
    Compute evaluation metrics including IoU, Dice, ROC AUC, and Precision-Recall AUC.
    Args:
        
        '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    # Compute IoU and Dice
    iou = metric_iou.compute(predictions=predictions, references=labels, num_labels=num_labels)
    dice = metric_dice.compute(predictions=predictions, references=labels)
    
    # Compute ROC and Precision-Recall curves
    roc_metrics = {}
    pr_metrics = {}
    logits_flat = logits.reshape(-1, num_labels)
    labels_flat = labels.reshape(-1)
    probs = torch.softmax(torch.tensor(logits_flat), dim=-1).numpy()
    
    for label in range(num_labels):
        if label in labels_flat:
            fpr, tpr, _ = roc_curve(labels_flat == label, probs[:, label])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(labels_flat == label, probs[:, label])
            pr_auc = auc(recall, precision)
            roc_metrics[f"roc_auc_{id2label[label]}"] = roc_auc
            pr_metrics[f"pr_auc_{id2label[label]}"] = pr_auc
            
            # Plot ROC and PR curves
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve: {id2label[label]}")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve: {id2label[label]}")
            plt.legend()
            
            plt.savefig(f"{output_dir}/{id2label[label]}_curves.png")
            plt.close()
    
    return {
        "mean_iou": iou["mean_iou"],
        "dice": dice["dice"],
        **roc_metrics,
        **pr_metrics
    }

def infer_and_visualize(image_path, output_path=None):
    '''
    Inference and visualization
    '''
    image = Image.open(image_path).convert("RGB")
    image = np.array(Image.fromarray(np.array(image)).resize((1000, 512), resample=Image.BILINEAR))
    inputs = image_processor(images=image, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_mask = outputs.logits.argmax(dim=1).squeeze().numpy()
    
    # Visualize
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image), cmap="gray")
    plt.title("Input OCT Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap="viridis")
    plt.title("Predicted Segmentation")
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return predicted_mask


#%% Class Labels & Colors
id2label = {0: "background", 1: "ILM", 2: "RPE", 3: "BM"}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# Define colors for annotations (RGBA, updated from user output)
color_map = {
    (255, 0, 0, 255): 1,    # ILM: Red
    (0, 0, 252, 255): 2,   # RPE:  Blue (252)
    (0, 0, 193, 255): 3     # BM: Blue (193)
}




########
# %% Sample Dataset 
image_paths, mask_paths = create_smaller_dataset(image_dir, mask_dir)
# limiting to 300 images for testing
image_paths = image_paths[:300]
mask_paths = mask_paths[:300]

# %% Split Dataset
train_images, temp_images, train_masks, temp_masks = train_test_split(
                                                    image_paths, mask_paths, test_size=0.2, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(
                                                    temp_images, temp_masks, test_size=0.5, random_state=42)



# %%  Create Datasets
trn_d  = create_dataset(train_images, train_masks)
val_d  = create_dataset(val_images, val_masks)
tst_d  = create_dataset(test_images, test_masks)
all_d  = DatasetDict({"trn": trn_d, "val": val_d, "tst": tst_d})



# %% Preprocess Data 
trn_d   = trn_d.map(preprocess_data, batched=True, remove_columns=["image", "label"])
val_d   = val_d.map(preprocess_data, batched=True, remove_columns=["image", "label"])
tst_d   = tst_d.map(preprocess_data, batched=True, remove_columns=["image", "label"])


# %% Load model
model = SegformerForSemanticSegmentation.from_pretrained("PekingU/rtdetr_v2_r18vd",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
            )

# %% Training Setup
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=6e-5,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    push_to_hub=False,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trn_d,
    eval_dataset=val_d,
    compute_metrics=compute_metrics
)

# %% Train the model
trainer.train()
trainer.save_model(f"{output_dir}-final")

# %%
plt.savefig(f"{output_dir}/{id2label[label]}_curves.png")

# %% Define metrics
 #metric_iou = load_metric("mean_iou")
 #metric_dice = load_metric("dice")
 print({
    "mean_iou": iou["mean_iou"],
    "dice": dice["dice"],
    **roc_metrics,
    **pr_metrics
})


# %% Example usage
infer_and_visualize("/home/suraj/Data/Duke_WLOA_RL_Annotated/AMD/output/image/AMD_001_045.png", "./segformer-oct-final/sample_prediction.png")


'''
# %% MAIN
def main():
    
    # Sample Dataset 
    image_paths, mask_paths = create_smaller_dataset(image_dir, mask_dir)
    # limiting to 300 images for testing
    image_paths = image_paths[:300]
    mask_paths = mask_paths[:300]

    # Split Dataset
    train_images, temp_images, train_masks, temp_masks = train_test_split(
                                                        image_paths, mask_paths, test_size=0.2, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(
                                                        temp_images, temp_masks, test_size=0.5, random_state=42)


    # Create Datasets
    trn_d  = create_dataset(train_images, train_masks)
    val_d  = create_dataset(val_images, val_masks)
    tst_d  = create_dataset(test_images, test_masks)
    all_d  = DatasetDict({"trn": trn_d, "val": val_d, "tst": tst_d})



    # Preprocess Data 
    trn_d   = trn_d.map(preprocess_data, batched=True, remove_columns=["image", "label"])
    val_d   = val_d.map(preprocess_data, batched=True, remove_columns=["image", "label"])
    tst_d   = tst_d.map(preprocess_data, batched=True, remove_columns=["image", "label"])


    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained("PekingU/rtdetr_v2_r18vd",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
                )
    
        # %% Training Setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        push_to_hub=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trn_d,
        eval_dataset=val_d,
        compute_metrics=compute_metrics
    )

    # %% Train the model
    trainer.train()
    trainer.save_model(f"{output_dir}-final")
    return None
    
    
if __name__ == "__main__":
    main()
    
    '''
    
## CHANGELOG