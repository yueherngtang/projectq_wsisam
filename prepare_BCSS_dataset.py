import cv2
import numpy as np
import os
import json
import re
import math
import traceback
import random

# --- CONFIGURATION ---
BASE_DIR = '/Volumes/1TB HDD 02/BCSS' 
IMAGE_DIR = os.path.join(BASE_DIR, 'data/images')
MASK_DIR = os.path.join(BASE_DIR, 'data/masks')
JSON_DIR = os.path.join(BASE_DIR, 'data/annotations')
OUTPUT_DIR = '/Volumes/1TB HDD 02/BCSSAugSplit'

OUTPUT_PATCH_DIR = '/Volumes/1TB HDD 02/BCSSPATCHAugSplit'

PATCH_SIZE_40X = 1024 
STRIDE = 512

CROP_SIZE_FOR_20X = 2048 
PATCH_SIZE_20X_FINAL = 1024

# Set Random Seed for Reproducibility
random.seed(42)

# Global list to store JSON records
DATA_RECORDS = [] 

# Ensure Directories Exist
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)

SPLITS = ['train', 'test']
CATEGORIES = ['images_40x', 'masks_40x', 'images_20x', 'masks_20x']
for split in SPLITS:
    for sub in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATCH_DIR, split, sub), exist_ok=True)

# --- HELPER FUNCTIONS ---

def parse_filename_coords(filename):
    """Extracts xmin and ymin from filename."""
    x_match = re.search(r'xmin(\d+)', filename)
    y_match = re.search(r'ymin(\d+)', filename)
    if x_match and y_match:
        return int(x_match.group(1)), int(y_match.group(1))
    return 0, 0

def load_json_robust(json_path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(json_path, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    print(f"Error: Failed to decode JSON {os.path.basename(json_path)}")
    return None

def get_axis_coordinates(total_len, patch_len, stride):
    if total_len < patch_len: return []
    coords = list(range(0, total_len - patch_len + 1, int(stride)))
    if (total_len - patch_len) - coords[-1] > (stride / 2.0):
        coords.append(total_len - patch_len)
    return coords

def robust_crop_padded(img, center_x, center_y, crop_size, is_mask=False):
    h, w = img.shape[:2]
    half_size = crop_size // 2
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size
    
    pad_left = abs(min(0, x1))
    pad_top = abs(min(0, y1))
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(w, x2)
    valid_y2 = min(h, y2)
    
    valid_crop = img[valid_y1:valid_y2, valid_x1:valid_x2]
    
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        val = 0 if is_mask else (240, 240, 240)
        return cv2.copyMakeBorder(valid_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=val)
    return valid_crop

def get_augmentations(img, mask):
    augs = {}
    augs['orig'] = (img, mask)
    augs['flipH'] = (cv2.flip(img, 1), cv2.flip(mask, 1))
    augs['flipV'] = (cv2.flip(img, 0), cv2.flip(mask, 0))
    augs['rot90'] = (cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augs['rot180'] = (cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180))
    augs['rot270'] = (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))
    return augs

# --- STAGE 1 LOGIC: CROP ROI FROM SLIDE ---

def process_roi_onestep(image_path, mask_path, json_path):
    data = load_json_robust(json_path)
    if data is None: return None

    try:
        if isinstance(data, list): roi = data[0]['annotation']['elements'][0]
        else: roi = data['annotation']['elements'][0]
        
        center_global = roi['center'] 
        target_w = int(roi['width'])
        target_h = int(roi['height'])
        rotation_rad = float(roi.get('rotation', 0))
    except Exception as e:
        print(f"Skipping {os.path.basename(json_path)}: {e}")
        return None

    xmin_patch, ymin_patch = parse_filename_coords(os.path.basename(image_path))
    local_cx = float(center_global[0] - xmin_patch)
    local_cy = float(center_global[1] - ymin_patch)

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    angle_deg = math.degrees(rotation_rad)
    M = cv2.getRotationMatrix2D((local_cx, local_cy), angle_deg, 1.0)
    
    M[0, 2] += (target_w / 2.0) - local_cx
    M[1, 2] += (target_h / 2.0) - local_cy

    crop_img = cv2.warpAffine(img, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    crop_mask = cv2.warpAffine(mask, M, (target_w, target_h), flags=cv2.INTER_NEAREST)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    binary_mask = np.where(crop_mask == 1, 255, 0).astype(np.uint8)

    return base_name, crop_img, binary_mask

def cut_bounding_box():
    """
    Stage 1: Reads raw data, crops ROIs based on JSON, saves them to disk.
    Does NOT generate patches.
    """
    print("--- Stage 1: Cutting Bounding Boxes ---")
    json_map = {}
    for jf in os.listdir(JSON_DIR):
        if jf.endswith('.json'):
            tcga_id = jf[:12] 
            json_map[tcga_id] = os.path.join(JSON_DIR, jf)

    img_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.tif', '.jpg'))]
    
    for img_file in img_files:
        tcga_id = img_file[:12] 
        if tcga_id in json_map:
            mask_candidates = [img_file.replace('image', 'mask'), img_file, os.path.splitext(img_file)[0] + '_mask.png']
            mask_path = None
            for cand in mask_candidates:
                if os.path.exists(os.path.join(MASK_DIR, cand)):
                    mask_path = os.path.join(MASK_DIR, cand)
                    break
            
            if mask_path:
                result = process_roi_onestep(
                    os.path.join(IMAGE_DIR, img_file), 
                    mask_path, 
                    json_map[tcga_id]
                )
                
                if result:
                    base_name, crop_img, binary_mask = result
                    # Save Stage 1 Output
                    save_path_im = os.path.join(OUTPUT_DIR, 'images', base_name + '.png')
                    save_path_mk = os.path.join(OUTPUT_DIR, 'masks', base_name + '.png')
                    cv2.imwrite(save_path_im, crop_img)
                    cv2.imwrite(save_path_mk, binary_mask)
                    print(f"Saved ROI: {base_name}")
    print("--- Stage 1 Complete ---\n")

# --- STAGE 2 LOGIC: PATCHING & AUGMENTATION ---

def generate_multires_patches(roi_img, roi_mask, base_name, split_type): 
    img_h, img_w = roi_img.shape[:2]
    xs = get_axis_coordinates(img_w, PATCH_SIZE_40X, STRIDE)
    ys = get_axis_coordinates(img_h, PATCH_SIZE_40X, STRIDE)
    
    count = 0
    for y in ys:
        for x in xs:
            # 1. Extract Raw Patches
            patch_40x_raw = roi_img[y:y+PATCH_SIZE_40X, x:x+PATCH_SIZE_40X]
            mask_40x_raw = roi_mask[y:y+PATCH_SIZE_40X, x:x+PATCH_SIZE_40X]
            
            cx = x + PATCH_SIZE_40X // 2
            cy = y + PATCH_SIZE_40X // 2
            patch_20x_large = robust_crop_padded(roi_img, cx, cy, CROP_SIZE_FOR_20X, is_mask=False)
            mask_20x_large = robust_crop_padded(roi_mask, cx, cy, CROP_SIZE_FOR_20X, is_mask=True)
            
            patch_20x_raw = cv2.resize(patch_20x_large, (PATCH_SIZE_20X_FINAL, PATCH_SIZE_20X_FINAL), interpolation=cv2.INTER_AREA)
            mask_20x_raw = cv2.resize(mask_20x_large, (PATCH_SIZE_20X_FINAL, PATCH_SIZE_20X_FINAL), interpolation=cv2.INTER_NEAREST)

            # 2. Augmentations
            if split_type == 'train':
                augs_40x = get_augmentations(patch_40x_raw, mask_40x_raw)
                augs_20x = get_augmentations(patch_20x_raw, mask_20x_raw)
            else:
                augs_40x = {'orig': (patch_40x_raw, mask_40x_raw)}
                augs_20x = {'orig': (patch_20x_raw, mask_20x_raw)}
            
            # 3. Save & Log
            for aug_key in augs_40x.keys():
                img_40, msk_40 = augs_40x[aug_key]
                img_20, msk_20 = augs_20x[aug_key]
                
                suffix = "" if aug_key == 'orig' else f"_{aug_key}"
                unique_name = f"{base_name}_x{x}_y{y}{suffix}"
                filename = f"{unique_name}.png"

                path_im_40 = os.path.join(OUTPUT_PATCH_DIR, split_type, 'images_40x', filename)
                path_mk_40 = os.path.join(OUTPUT_PATCH_DIR, split_type, 'masks_40x', filename)
                path_im_20 = os.path.join(OUTPUT_PATCH_DIR, split_type, 'images_20x', filename)
                path_mk_20 = os.path.join(OUTPUT_PATCH_DIR, split_type, 'masks_20x', filename)

                cv2.imwrite(path_im_40, img_40)
                cv2.imwrite(path_mk_40, msk_40)
                cv2.imwrite(path_im_20, img_20)
                cv2.imwrite(path_mk_20, msk_20)

                DATA_RECORDS.append({
                    "split": split_type,
                    "name": unique_name,
                    "augmentation": aug_key,
                    "high": {"im_path": path_im_40, "gt_path": path_mk_40},
                    "low": {"im_path": path_im_20, "gt_path": path_mk_20}
                })
                count += 1
    return count

def process_saved_rois_to_patches():
    """
    Stage 2: Reads the saved ROIs from Stage 1, Splits them, Augments them, 
    and generates patches and the JSON registry.
    """
    print("--- Stage 2: Generating Patches & Splitting ---")
    
    # Input for Stage 2 is Output of Stage 1
    roi_image_dir = os.path.join(OUTPUT_DIR, 'images')
    roi_mask_dir = os.path.join(OUTPUT_DIR, 'masks')

    if not os.path.exists(roi_image_dir):
        print("Error: No saved ROIs found. Run cut_bounding_box() first.")
        return

    roi_files = [f for f in os.listdir(roi_image_dir) if f.endswith('.png')]
    print(f"Found {len(roi_files)} ROIs to process.")

    processed = 0
    for filename in roi_files:
        img_path = os.path.join(roi_image_dir, filename)
        mask_path = os.path.join(roi_mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask missing for {filename}, skipping.")
            continue
        
        # Load ROI
        roi_img = cv2.imread(img_path)
        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # ### Random Split Decision (Per ROI)
        split_type = 'train' if random.random() < 0.70 else 'test'
        
        base_name = os.path.splitext(filename)[0]
        
        # Generate patches
        count = generate_multires_patches(roi_img, roi_mask, base_name, split_type)
        print(f"Processed {base_name}: {count} patches ({split_type})")
        processed += 1

    # Save JSON
    json_output_path = os.path.join(OUTPUT_PATCH_DIR, "dataset.json")
    print(f"Saving dataset registry to {json_output_path}...")
    with open(json_output_path, "w") as f:
        json.dump(DATA_RECORDS, f, indent=4)
    print(f"--- Stage 2 Complete --- Processed {processed} ROIs.")

if __name__ == "__main__":
    # Uncomment the function you want to run!
    
    # STEP 1: Run this once to create the clean ROIs
    #cut_bounding_box()
    
    # STEP 2: Run this to generate patches/augmentation/json
    process_saved_rois_to_patches()

            
# IMAGE_DIR = '/Volumes/1TB HDD 02/BCSS/data/images'
# MASK_DIR = '/Volumes/1TB HDD 02/BCSS/data/masks'
# OUTPUT_DIR = '/Volumes/1TB HDD 02/test_BCSS2'

# os.makedirs(IMAGE_DIR, exist_ok=True)
# os.makedirs(MASK_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def process_tumor_mask(image_path, mask_path, output_path):
#     """
#     Reads a mask, converts the tumor area (value=1) to White (255) 
#     and everything else to Black (0), and saves the result.
#     """
#     # 1. Read the mask in grayscale to get raw pixel values
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         print(f"Error: Could not read mask {mask_path}")
#         return

#     # 2. Create the Binary Black & White Mask
#     # Logic: If pixel is 1 (Tumor), make it 255 (White). Else make it 0 (Black).
#     binary_bw_mask = np.where(mask == 1, 255, 0).astype(np.uint8)

#     # 3. Save the Binary Mask
#     # We prefix with 'bw_' to distinguish it
#     base_name = os.path.basename(image_path)
#     # Ensure the output is saved as .png to prevent compression artifacts on the mask
#     save_name = 'bw_mask_' + os.path.splitext(base_name)[0] + '.png'
#     binary_output_filename = os.path.join(output_path, save_name)
    
#     cv2.imwrite(binary_output_filename, binary_bw_mask)
#     print(f"Saved Binary Mask: {binary_output_filename}")

#     # --- OPTIONAL: Overlay Visualization (For debugging) ---
#     # If you want to visually verify the mask matches the tissue, uncomment below:

#     image_bgr = cv2.imread(image_path)
#     if image_bgr is not None:
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         overlay = image_rgb.copy()
        
#         # Make tumor area Red
#         tumor_bool = (mask == 1)
#         red_color = np.array([255, 0, 0], dtype=np.uint8)
#         alpha = 0.5
        
#         for c in range(3):
#             overlay[tumor_bool, c] = (alpha * red_color[c] + (1 - alpha) * overlay[tumor_bool, c])
            
#         overlay_name = 'overlay_' + base_name
#         cv2.imwrite(os.path.join(output_path, overlay_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# # --- Main processing loop ---
# print("\nStarting processing...")

# # Get list of valid image extensions
# valid_extensions = ('.png', '.jpg', '.tif', '.jpeg')

# for filename in os.listdir(IMAGE_DIR):
#     if filename.lower().endswith(valid_extensions):
#         image_file_path = os.path.join(IMAGE_DIR, filename)
        
#         # --- File Matching Logic ---
#         # Adjust this logic if your filenames differ. 
#         # Example: 'train_1.png' -> 'train_1_mask.png' or similar.
#         # The current logic assumes 'image' is in the name and replaces it with 'mask'.
#         mask_filename = filename.replace('image', 'mask')
        
#         # Fallback: if filename doesn't contain 'image', try simply appending or matching ID
#         # (You can customize this block if your naming convention is different)
        
#         mask_file_path = os.path.join(MASK_DIR, mask_filename)

#         if os.path.exists(mask_file_path):
#             process_tumor_mask(image_file_path, mask_file_path, OUTPUT_DIR)
#         else:
#             # Try one more common convention: same name but in mask folder
#             mask_file_path_direct = os.path.join(MASK_DIR, filename)
#             if os.path.exists(mask_file_path_direct):
#                 process_tumor_mask(image_file_path, mask_file_path_direct, OUTPUT_DIR)
#             else:
#                 print(f"Skipping {filename}: Mask not found (Checked {mask_filename} and {filename})")

# print("\nProcessing complete. Check the output directory.")