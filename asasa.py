import numpy as np
from PIL import Image

# Load your mask
img = np.array(Image.open('/Volumes/1TB HDD 01/CATCH4WSISAM(withHard-ve&1024)/high/masks/Histiocytoma_01_1_2_edge.png').convert("L"))

print(f"Data Type: {img.dtype}")
print(f"Unique values: {np.unique(img)}")
print(f"Max value: {img.max()}")

gt_mask_np = (img > 0).astype(float)

print(f"After Binarization - Unique values: {np.unique(gt_mask_np)}")
print(f"After Binarization - Max value: {gt_mask_np.max()}")
