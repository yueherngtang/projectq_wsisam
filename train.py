import os
# Allow PyTorch to use CPU for operations missing on MPS (Fixes grid_sampler error)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import tqdm
import torch
import numpy as np
import cv2
import json
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Imports
from segment_anything_training import sam_model_registry
from segment_anything_training.utils.transforms import ResizeLongestSide
from network import MaskDecoderHigh, MaskDecoderLow
from utils.loss_mask import loss_masks

# ================= CONFIGURATION =================
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
DATASET_JSON = "/data/vai_aimed/BCSS/BCSSPATCHAugSplit/dataset.json"
CHECKPOINT_DIR = "/data/vai_aimed/code/projectq_wsisam/checkpoint/BCSS_1"
PRETRAINED_SAM = "/data/vai_aimed/code/projectq_wsisam/mobile_sam.pt"

# This overwrites PRETRAINED_SAM, put None if you don't want to load previous weights
PREVIOUS_CKPT_DIR = None
RESUME_EPOCH = 4 # Epoch number of the previous checkpoint to load

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 15
IMG_SIZE = 1024
LAMBDA = 0.5 
ACCUMULATION_STEPS = 4
# =================================================

class WSIPairedDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        self.data = [d for d in all_data if d.get('split') == 'train']
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        print(f"Loaded {len(self.data)} training samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def preprocess(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def process_pair(self, item, key):
        img = cv2.imread(item[key]['im_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.tensor(img).permute(2, 0, 1).float()
        x = self.preprocess(x)

        mask = cv2.imread(item[key]['gt_path'], 0)
        mask = (mask > 128).astype(np.float32)
        y = torch.tensor(mask).unsqueeze(0).float()
        return x, y

    def __getitem__(self, idx):
        item = self.data[idx]
        img_h, mask_h = self.process_pair(item, "high")
        img_l, mask_l = self.process_pair(item, "low")
        return {"image_high": img_h, "mask_high": mask_h, "image_low": img_l, "mask_low": mask_l}

def get_random_prompts(mask_gt):

    y_indices, x_indices = np.where(mask_gt.squeeze().cpu().numpy() == 1)
    if len(x_indices) == 0:
        return (torch.tensor([[[0, 0]]]).float().to(DEVICE), torch.tensor([[0]]).float().to(DEVICE))
    idx = np.random.choice(len(x_indices), 1)[0]
    return (torch.tensor([[[x_indices[idx], y_indices[idx]]]]).float().to(DEVICE), torch.tensor([[1]]).float().to(DEVICE))

def get_box_prompt(mask_gt):
    y_indices, x_indices = np.where(mask_gt.squeeze().cpu().numpy() == 1)
    
    # CASE 1: EMPTY MASK (Hard Negative)
    if len(x_indices) == 0:
        # Must use full box, as there is nothing else to prompt
        box = torch.tensor([[0.0, 0.0, 1024.0, 1024.0]]).to(DEVICE)
        return box

    # CASE 2: POSITIVE MASK
    
    # --- CRITICAL FIX: PROMPT AUGMENTATION ---
    # 50% chance to simulate "Inference Mode" (Full Box)
    # This teaches the model: "Even if the box is huge, look for the tumor inside!"
    if np.random.rand() < 0.5:
        box = torch.tensor([[0.0, 0.0, 1024.0, 1024.0]]).to(DEVICE)
        return box
    # -----------------------------------------

    # The other 50% of the time, use the Tight Box 
    # (This helps the model learn what the tumor features actually look like)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add noise
    noise = 20 # Increased noise slightly
    x_min = max(0, x_min - np.random.randint(0, noise))
    y_min = max(0, y_min - np.random.randint(0, noise))
    x_max = min(1024, x_max + np.random.randint(0, noise))
    y_max = min(1024, y_max + np.random.randint(0, noise))

    box = torch.tensor([[float(x_min), float(y_min), float(x_max), float(y_max)]]).to(DEVICE)
    return box


# Helper to get intermediate features from backbone
def get_features(sam_model, x):
    # CRITICAL: TinyViT expects 1024x1024. 
    # If we pass 256, we must resize first.
    if x.shape[-1] != 1024:
        x_1024 = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
    else:
        x_1024 = x

    image_embeddings = sam_model.image_encoder(x_1024)
    
    # Try to access intermediate features
    # Note: If your image_encoder.py doesn't return the tuple, this block handles the extraction manually
    # if 'sam.image_encoder.backbone' is accessible.
    interm_embeddings = None
    try:
        if hasattr(sam_model.image_encoder, 'forward_features'):
             # This assumes you modified image_encoder to return tuple
             # If not, this might return just embeddings. 
             # Check if result is tuple:
             if isinstance(image_embeddings, tuple):
                 image_embeddings, interm_embeddings = image_embeddings
        
        if interm_embeddings is None and hasattr(sam_model.image_encoder, 'backbone'):
             # Fallback: Extract from backbone directly if possible
             # This runs the backbone again, which is inefficient, but safe if API missing
             features = sam_model.image_encoder.backbone(x_1024)
             interm_embeddings = features[-1]
             
        if interm_embeddings is None:
             # Last resort fallback (Use embeddings as intermediate - strictly a placeholder)
             print("Warning: Using embeddings as intermediate features (Suboptimal).")
             interm_embeddings = image_embeddings

    except Exception as e:
        print(f"Feature extraction warning: {e}")
        interm_embeddings = image_embeddings 

    return image_embeddings, interm_embeddings


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dataset = WSIPairedDataset(DATASET_JSON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2,           # Use 2 parallel processes for loading
                        persistent_workers=True, # Keep workers alive between epochs (Critical for speed)
                        pin_memory=True          # Faster transfer to GPU/MPS
    )

    if PREVIOUS_CKPT_DIR is not None:
        print("Loading Pre-trained Backbone...")
        # 1. Initialize empty SAM
        sam = sam_model_registry["vit_tiny"](checkpoint=None).to(DEVICE)
        
        # 2. Load your FINE-TUNED backbone weights
        #sam_ckpt = f"{PREVIOUS_CKPT_DIR}/sam_backbone_ep{RESUME_EPOCH}.pth"
        sam_ckpt = f"{PRETRAINED_SAM}"
        print(f"Loading backbone from: {sam_ckpt}")
        sam.load_state_dict(torch.load(sam_ckpt, map_location=DEVICE))
        
        # Ensure it's still trainable
        for p in sam.parameters(): p.requires_grad = True 
        
        # 3. Initialize Decoders
        model_high = MaskDecoderHigh(model_type="vit_tiny").to(DEVICE).train()
        model_low = MaskDecoderLow(model_type="vit_tiny").to(DEVICE).train()
        
        # 4. Load your FINE-TUNED decoder weights
        high_ckpt = f"{PREVIOUS_CKPT_DIR}/high_ep{RESUME_EPOCH}.pth"
        low_ckpt = f"{PREVIOUS_CKPT_DIR}/low_ep{RESUME_EPOCH}.pth"
        
        print(f"Loading decoders from: {high_ckpt} and {low_ckpt}")
        model_high.load_state_dict(torch.load(high_ckpt, map_location=DEVICE))
        model_low.load_state_dict(torch.load(low_ckpt, map_location=DEVICE))
    
    else:
        print("Loading Backbone...")
        sam = sam_model_registry["vit_tiny"](checkpoint=PRETRAINED_SAM).to(DEVICE)
        # Unfreeze backbone for optimal results
        for p in sam.parameters(): p.requires_grad = True 
        
        model_high = MaskDecoderHigh(model_type="vit_tiny").to(DEVICE).train()
        model_low = MaskDecoderLow(model_type="vit_tiny").to(DEVICE).train()

    # for name, param in sam.named_parameters():
    #     if "image_encoder" in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True # Keep prompt_encoder trainable!

    # Update optimizer to only include trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, 
               list(sam.parameters()) + list(model_high.parameters()) + list(model_low.parameters())), 
        lr=LEARNING_RATE
    )
    
    # optimizer = AdamW(
    #     list(sam.parameters()) + list(model_high.parameters()) + list(model_low.parameters()), 
    #     lr=LEARNING_RATE
    # )
    # optimizer = AdamW(
    #     list(model_high.parameters()) + list(model_low.parameters()), 
    #     lr=LEARNING_RATE
    # )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"Starting Training on {DEVICE}...")

    epoch_loss_list = []
    lr_list =[]

    # ... (Setup code remains the same) ...

    for epoch in range(EPOCHS):
        epoch_loss = 0

        running_pos_loss = 0.0
        running_neg_loss = 0.0
        running_pos_count = 0
        running_neg_count = 0

        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            img_h, mask_h = batch['image_high'].to(DEVICE), batch['mask_high'].to(DEVICE)
            img_l, mask_l = batch['image_low'].to(DEVICE), batch['mask_low'].to(DEVICE)

            # 1. Features (Batch compatible)
            emb_h, interm_h = get_features(sam, img_h)
            emb_l, interm_l = get_features(sam, img_l)
            
            # with torch.no_grad():
            #     pts_h_list, lbl_h_list = [], []
            #     pts_l_list, lbl_l_list = [], []
                
            #     # Iterate over the batch dimension
            #     for b in range(img_h.shape[0]):
            #         # Extract single mask: (1, 1024, 1024)
            #         curr_mask_h = mask_h[b] 
            #         curr_mask_l = mask_l[b]
                    
            #         # Generate prompt for this specific image
            #         p_h, l_h = get_random_prompts(curr_mask_h)
            #         p_l, l_l = get_random_prompts(curr_mask_l)
                    
            #         pts_h_list.append(p_h)
            #         lbl_h_list.append(l_h)
            #         pts_l_list.append(p_l)
            #         lbl_l_list.append(l_l)
                
            #     # Stack back into batch: (B, 1, 2)
            #     pts_h = torch.cat(pts_h_list, dim=0)
            #     lbl_h = torch.cat(lbl_h_list, dim=0)
            #     pts_l = torch.cat(pts_l_list, dim=0)
            #     lbl_l = torch.cat(lbl_l_list, dim=0)

            #     # Batch Prompt Encoding
            #     sparse_h, dense_h = sam.prompt_encoder(points=(pts_h, lbl_h), boxes=None, masks=None)
            #     sparse_l, dense_l = sam.prompt_encoder(points=(pts_l, lbl_l), boxes=None, masks=None)

            with torch.no_grad():
                box_h_list = []
                box_l_list = []
                
                # Iterate over the batch dimension
                for b in range(img_h.shape[0]):
                    curr_mask_h = mask_h[b] 
                    curr_mask_l = mask_l[b]
                    
                    # Generate BOX prompt instead of Point prompt
                    b_h = get_box_prompt(curr_mask_h)
                    b_l = get_box_prompt(curr_mask_l)
                    
                    box_h_list.append(b_h)
                    box_l_list.append(b_l)
                
                # Stack into batch: Result shape should be (B, 1, 4)
                boxes_h = torch.cat(box_h_list, dim=0)
                boxes_l = torch.cat(box_l_list, dim=0)

                # CRITICAL CHANGE: 
                # 1. Pass 'boxes=boxes_h'
                # 2. Set 'points=None'
                sparse_h, dense_h = sam.prompt_encoder(points=None, boxes=boxes_h, masks=None)
                sparse_l, dense_l = sam.prompt_encoder(points=None, boxes=boxes_l, masks=None)

            # 3. Decoder

            current_batch_size = img_h.shape[0]
            dense_pe = sam.prompt_encoder.get_dense_pe()
            
            if dense_pe.shape[0] != current_batch_size:
                dense_pe = dense_pe.repeat(current_batch_size, 1, 1, 1)

            masks_sam_h, _, token_h, wsi_emb_h = model_high(
                image_embeddings=emb_h,
                image_pe=dense_pe, # <--- Pass the expanded PE here
                sparse_prompt_embeddings=sparse_h,
                dense_prompt_embeddings=dense_h,
                multimask_output=False,
                wsi_token_only=False,
                interm_embeddings=interm_h
            )
            
            masks_sam_l, _, token_l, wsi_emb_l = model_low(
                image_embeddings=emb_l,
                image_pe=dense_pe, # <--- And here
                sparse_prompt_embeddings=sparse_l,
                dense_prompt_embeddings=dense_l,
                multimask_output=False,
                wsi_token_only=False,
                interm_embeddings=interm_l
            )

            # --- FUSION ---
            token_fused = token_h + token_l
            
            B_curr, C, N = wsi_emb_h.shape
            H_emb = int(np.sqrt(N)) # 256
            
            # Reshape using current batch size B_curr
            pred_h_raw = (token_fused @ wsi_emb_h).view(B_curr, 1, H_emb, H_emb)
            pred_l_raw = (token_fused @ wsi_emb_l).view(B_curr, 1, H_emb, H_emb)

            pred_h_256 = pred_h_raw + masks_sam_h
            pred_l_256 = pred_l_raw + masks_sam_l

            pred_h_final = F.interpolate(pred_h_256, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred_l_final = F.interpolate(pred_l_256, size=(1024, 1024), mode='bilinear', align_corners=False)

            def calculate_batch_smart_loss(pred_batch, gt_batch):
                total_batch_loss = 0

                p_loss_sum = 0
                n_loss_sum = 0
                p_count = 0
                n_count = 0

                batch_size = pred_batch.shape[0]
                
                # Iterate per sample in batch
                for b in range(batch_size):
                    curr_pred = pred_batch[b].unsqueeze(0)
                    curr_gt = gt_batch[b].unsqueeze(0)
                    
                    # print(curr_pred)

                    if curr_gt.sum() == 0:
                        loss = F.binary_cross_entropy_with_logits(curr_pred, torch.zeros_like(curr_pred))
                        n_loss_sum += loss.item()
                        n_count += 1
                        total_batch_loss += (loss * 2.0)
                    else:
                        # POSITIVE PATCH
                        l_focal, l_dice = loss_masks(curr_pred, curr_gt, num_masks=1)
                        loss = l_focal + l_dice
                        p_loss_sum += loss.item()
                        p_count += 1
                        total_batch_loss +=  loss
                
                avg_loss = total_batch_loss / batch_size
                return avg_loss, p_loss_sum, p_count, n_loss_sum, n_count

            loss_h, ph_sum, ph_cnt, nh_sum, nh_cnt = calculate_batch_smart_loss(pred_h_final, mask_h)
            loss_l, pl_sum, pl_cnt, nl_sum, nl_cnt = calculate_batch_smart_loss(pred_l_final, mask_l)

            running_pos_loss += (ph_sum + pl_sum)
            running_pos_count += (ph_cnt + pl_cnt)
            running_neg_loss += (nh_sum + nl_sum)
            running_neg_count += (nh_cnt + nl_cnt)

            total_loss = (LAMBDA * loss_h) + ((1.0 - LAMBDA) * loss_l)
            loss_normalized = total_loss / ACCUMULATION_STEPS
            loss_normalized.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                avg_pos = running_pos_loss / (running_pos_count + 1e-6)
                avg_neg = running_neg_loss / (running_neg_count + 1e-6)

                print(f"Step {i}: Loss: {total_loss.item():.4f} | Pos Avg: {avg_pos:.4f} | Neg Avg: {avg_neg:.4f}")
            
                running_pos_loss = 0.0
                running_neg_loss = 0.0
                running_pos_count = 0
                running_neg_count = 0
                
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        epoch_loss_list.append(avg_loss)
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step(avg_loss)
        
        torch.save(model_high.state_dict(), f"{CHECKPOINT_DIR}/high_ep{epoch+1}.pth")
        torch.save(model_low.state_dict(), f"{CHECKPOINT_DIR}/low_ep{epoch+1}.pth")
        torch.save(sam.state_dict(), f"{CHECKPOINT_DIR}/sam_backbone_ep{epoch+1}.pth")

    print("Training Complete.")
    print("Epoch Losses:", epoch_loss_list)
    print("Learning Rates:", lr_list)

if __name__ == "__main__":
    main()

