import argparse
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from PIL import Image

# Import logic from your existing inference script (assumed to be named inference.py)
# If your previous file is named something else, change 'inference' to that filename.
try:
    from inference import load_model, run_inference, load_image
except ImportError:
    print("Error: Could not import from 'inference.py'. Please ensure your previous code is saved as 'inference.py' in the same directory.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WSI-SAM model performance on a dataset.")
    
    # Dataset arguments
    parser.add_argument("--dataset_dir", type=str, default= '/Volumes/1TB HDD 01/CATCH4WSISAM(withHard-ve&1024)2', help="Root directory of the testing dataset.")
    parser.add_argument("--high_res_folder", type=str, default="high/imgs", help="Subfolder name for high-res images.")
    parser.add_argument("--low_res_folder", type=str, default="low/imgs", help="Subfolder name for low-res images.")
    parser.add_argument("--mask_folder", type=str, default="high/masks", help="Subfolder name for ground truth masks.")
    
    # Model arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model.")
    parser.add_argument("--model_type", type=str, default="vit_tiny", help="Model type for SAM.")
    parser.add_argument("--checkpoint_SAM", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/checkpoint/trained_checkpoints_b2_1e4schedular_e8_1024x1024_smartloss2/sam_backbone_ep3.pth', help="Path to SAM checkpoint.")
    parser.add_argument("--checkpoint_net_high", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/checkpoint/trained_checkpoints_b2_1e4schedular_e8_1024x1024_smartloss2/high_ep3.pth', help="Path to High-Res Net checkpoint.")
    parser.add_argument("--checkpoint_net_low", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/checkpoint/trained_checkpoints_b2_1e4schedular_e8_1024x1024_smartloss2/low_ep3.pth', help="Path to Low-Res Net checkpoint.")
    
    # Inference arguments
    parser.add_argument("--bounding_box", type=str, default="[0, 0, 1024, 1024]", help="Bounding box 'xmin,ymin,xmax,ymax'.")
    
    return parser.parse_args()

def calculate_cancer_metrics(pred_mask, gt_mask):
    """
    Calculates Dice, IoU, Precision, Recall, F1, and Hausdorff Distance.
    Args:
        pred_mask (np.array): Binary prediction mask (0 or 1).
        gt_mask (np.array): Binary ground truth mask (0 or 1).
    """
    # Flatten for set metrics
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)

    #print(f"Calculating metrics: Pred unique values {np.unique(pred_mask)}, GT unique values {np.unique(gt_mask)}")

    # Intersection and Union
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    
    # True Positives, False Positives, False Negatives
    tp = intersection
    fp = np.logical_and(pred_flat, ~gt_flat).sum()
    fn = np.logical_and(~pred_flat, gt_flat).sum()

    # 1. Dice / F1 Score (They are mathematically equivalent for binary classification)
    dice = (2. * intersection) / (pred_flat.sum() + gt_flat.sum() + 1e-8)
    f1 = dice 

    # 2. IoU
    iou = intersection / (union + 1e-8)

    # 3. Precision
    precision = tp / (tp + fp + 1e-8)

    # 4. Recall
    recall = tp / (tp + fn + 1e-8)

    # # 5. Hausdorff Distance, why so slow omg
    # # HD is distance between boundaries. If masks are empty, HD is undefined (inf).
    # if np.any(pred_mask) and np.any(gt_mask):
    #     # directed_hausdorff returns (dist, index_1, index_2)
    #     d_pred_gt = directed_hausdorff(np.argwhere(pred_mask), np.argwhere(gt_mask))[0]
    #     d_gt_pred = directed_hausdorff(np.argwhere(gt_mask), np.argwhere(pred_mask))[0]
    #     hd = max(d_pred_gt, d_gt_pred)
    # else:
    #     # Penalize if one is empty and other isn't, or 0 if both empty
    #     if np.any(pred_mask) != np.any(gt_mask):
    #         hd = 100.0 # Arbitrary high penalty
    #     else:
    #         hd = 0.0

    return {
        "Dice": dice,
        "IoU": iou,
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    }

def calculate_healthy_metrics(pred_mask, gt_mask):
    """ Metrics for images WITHOUT cancer (Negative Samples) """
    # GT is All Black (0). 
    # Any white pixel in Pred is a False Positive.
    pred_bool = pred_mask > 0.5
    
    # True Negatives: The number of black pixels correctly predicted as black
    tn = (~pred_bool).sum()
    # False Positives: The number of white pixels (Hallucinations)
    fp = pred_bool.sum()
    # Total pixels
    total = pred_bool.size
    
    # 1. Specificity (True Negative Rate)
    # Target: 1.0
    specificity = tn / (total + 1e-8)
    
    # 2. False Positive Rate (FPR)
    # Target: 0.0
    fpr = fp / (total + 1e-8)
    
    # 3. Clean (Did it predict EMPTY?)
    # 1.0 if perfectly empty, 0.0 if any noise
    is_clean = 1.0 if fp == 0 else 0.0

    # 1. Invert the masks (Flip 0 to 1, and 1 to 0)
    # logic: 1 - 0 = 1 (New Target), 1 - 1 = 0 (New Background)
    pred_inv = 1 - pred_mask
    gt_inv = 1 - gt_mask
    
    # 2. Convert to Boolean for set operations
    pred_flat = pred_inv.flatten().astype(bool)
    gt_flat = gt_inv.flatten().astype(bool)

    # 3. Calculate Metrics on the INVERTED masks
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    
    # TP here means "Correctly predicted background"
    tp = intersection
    fp = np.logical_and(pred_flat, ~gt_flat).sum()
    fn = np.logical_and(~pred_flat, gt_flat).sum()

    # Dice (Now measures how well you covered the healthy tissue)
    dice = (2. * tp) / (pred_flat.sum() + gt_flat.sum() + 1e-8)
    
    # IoU
    iou = tp / (union + 1e-8)
    
    # Precision ("Of the pixels I called healthy, how many were actually healthy?")
    precision = tp / (tp + fp + 1e-8)
    
    # Recall ("Of all actual healthy pixels, how many did I find?")
    recall = tp / (tp + fn + 1e-8)


    return {"Specificity": specificity, "FPR": fpr,
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall}

def main():
    args = parse_args()
    
    # 1. Load Model
    print(f"Loading models on {args.device}...")
    # We pass 'args' to load_model, ensuring it has necessary checkpoint paths
    sam_model, net_high, net_low = load_model(args)
    
    # 2. Prepare File Lists
    high_dir = os.path.join(args.dataset_dir, args.high_res_folder)
    low_dir = os.path.join(args.dataset_dir, args.low_res_folder)
    mask_dir = os.path.join(args.dataset_dir, args.mask_folder)

    # Get all images (assuming .png, change extension if needed)
    image_files = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    
    if len(image_files) == 0:
        print(f"No images found in {high_dir}")
        return

    # Parse Bounding Box
    try:
        box = eval(args.bounding_box) # Converts string "[0,0,1024,1024]" to list
    except:
        print("Error parsing bounding box. Ensure format is '[x,y,w,h]'")
        box = [0, 0, 1024, 1024]

    # Metrics Accumulator
    cancer_stats = {"Dice": [], "IoU": [], "F1": [], "Precision": [], "Recall": []}
    healthy_stats = {"Specificity": [], "FPR": [], "Dice": [], "IoU": [], "Precision": [], "Recall": []}

    print(f"Starting inference on {len(image_files)} images...")
    
    for img_name in tqdm(image_files):
        # Construct paths
        high_path = os.path.join(high_dir, img_name)
        
        # Assumption: Low res and Mask have same filename. 
        # If they have different suffixes (e.g. _L.png), modify logic here.
        # Current logic: tries to match exact name, or fallback specific replacements if common
        low_path = os.path.join(low_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        # Fallback for naming conventions (Optional, based on your previous code snippet examples)
        if not os.path.exists(low_path):
             print(f"Warning: Image not found for Low {img_name}, skipping.")
             continue

        if not os.path.exists(high_path):
             print(f"Warning: Image not found for High {img_name}, skipping.")
             continue

        if not os.path.exists(mask_path):
             print(f"Warning: Mask not found for {img_name}, skipping.")
             continue
        
        # Load Data
        high_res_image = load_image(high_path)
        low_res_image = load_image(low_path)
        
        # Load Ground Truth Mask
        gt_mask_pil = Image.open(mask_path).convert("L") # Load as grayscale
        gt_mask = np.array(gt_mask_pil)
        gt_mask = (gt_mask > 0).astype(float) # Binarize Ground Truth (0.0 or 1.0)
        
        try:
            mask_logits, masks_wsi_low  = run_inference(
                high_res_image, 
                low_res_image, 
                box, 
                args, 
                test=True, 
                sam_model=sam_model, 
                net_high=net_high, 
                net_low=net_low
            )
        except TypeError:
            # Fallback if run_inference doesn't accept model objects as args
            print("Warning: run_inference does not accept model objects, using fallback which reloads models each time.")
            mask_logits, masks_wsi_low = run_inference(high_res_image, low_res_image, box, args, test=True)

        # Post-process Prediction
        mask_prob = torch.sigmoid(mask_logits)
        pred_mask = (mask_prob > 0.5).float().cpu().numpy().squeeze()
        
        # Resize GT to match Prediction if necessary (SAM outputs 1024x1024 usually)
        #print(f"Pred Mask Shape: {pred_mask.shape}, GT Mask Shape: {gt_mask.shape}")
        if pred_mask.shape != gt_mask.shape:
             # Resize pred_mask to match GT size (or vice versa)
             # usually better to evaluate at original resolution (GT size)
             print("Resizing predicted mask to match ground truth size for metric calculation.")
             import cv2
             pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        if gt_mask.sum() > 0:
            # === CANCER PATCH (Positive) ===
            res = calculate_cancer_metrics(pred_mask, gt_mask)
            for k, v in res.items(): cancer_stats[k].append(v)
        else:
            # === HEALTHY PATCH (Negative) ===
            res = calculate_healthy_metrics(pred_mask, gt_mask)
            for k, v in res.items(): healthy_stats[k].append(v)

    # 5. Print Split Report
    print("\n" + "="*45)
    print("      DATASET SPLIT REPORT      ")
    print("="*45)
    
    print(f"\n[SET 1] CANCER PATCHES (Count: {len(cancer_stats['Dice'])})")
    if len(cancer_stats['Dice']) > 0:
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
        print("-" * 40)
        for k, v in cancer_stats.items():
            print(f"{k:<15} | {np.mean(v):.4f}     | {np.std(v):.4f}")
    else:
        print("No cancer patches found.")

    print(f"\n[SET 2] HEALTHY PATCHES (Count: {len(healthy_stats['Specificity'])})")
    if len(healthy_stats['Specificity']) > 0:
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
        print("-" * 40)
        for k, v in healthy_stats.items():
            print(f"{k:<15} | {np.mean(v):.4f}     | {np.std(v):.4f}")
    else:
        print("No healthy patches found.")
    print("="*45)

if __name__ == "__main__":
    main()