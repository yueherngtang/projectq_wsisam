import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything_training import sam_model_registry
from network import MaskDecoderHigh, MaskDecoderLow
import torch.nn.functional as F
import os
from typing import Tuple
import copy

def parse_args():
    """
    Parses command-line arguments to retrieve image paths and bounding box input.
    """
    parser = argparse.ArgumentParser(description="Run inference with high-res and low-res images with a bounding box.")
    parser.add_argument("--high_res_image", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/WSI-SAM/examples/TCGA-AO-A0J4-DX1_xmin17194_ymin10629_MAG-40.00_x0_y0_H.png', help="Path to the high-resolution image.")
    parser.add_argument("--low_res_image", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/WSI-SAM/examples/TCGA-AO-A0J4-DX1_xmin17194_ymin10629_MAG-40.00_x0_y0_L.png', help="Path to the low-resolution image.")
    parser.add_argument("--bounding_box", type=str, default=[0, 0, 1024, 1024], help="Bounding box coordinates in the format 'xmin,ymin,xmax,ymax'.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model (default: cuda:0)")
    parser.add_argument("--visual", type=str, default=True, help="visualize the output")
    parser.add_argument("--model_type", type=str, default="vit_tiny", help="Model type for SAM")
    parser.add_argument("--checkpoint_SAM", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/checkpoint/trained_checkpoints_b2_1e4schedular_e8_1024x1024_smartloss2/sam_backbone_ep3.pth', help="Path to the trained model checkpoint")
    parser.add_argument("--checkpoint_net_high", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/checkpoint/trained_checkpoints_b2_1e4schedular_e8_1024x1024_smartloss2/high_ep3.pth', help="Path to net")
    parser.add_argument("--checkpoint_net_low", type=str, default='/Users/yueherngtang/Desktop/WSI-SAM/checkpoint/trained_checkpoints_b2_1e4schedular_e8_1024x1024_smartloss2/low_ep3.pth', help="Path to net1")
    args = parser.parse_args()
    return args

        

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([255, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def load_image(image_path: str) -> torch.Tensor:
    """
    Loads an image from a file path and returns it as a torch.Tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Loaded image as a (C, H, W) tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image_tensor = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)  # Convert to (C, H, W)
    return image_tensor

# def load_image(image_path: str) -> torch.Tensor:
#     """
#     Loads an image and applies the SAME preprocessing as training.
#     """
#     # 1. Load Image
#     image = Image.open(image_path).convert("RGB")
#     image = np.array(image)
    
#     # 2. Convert to Float Tensor (C, H, W)
#     # CRITICAL: Must be Float, not Uint8, for normalization
#     x = torch.tensor(image).permute(2, 0, 1).float() 
    
#     # 3. Define Normalization Constants (From your training script)
#     pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
#     # 4. Normalize
#     x = (x - pixel_mean) / pixel_std
    
#     return x


def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:

    masks = F.interpolate(
        masks,
        (1024,1024),
        mode="bilinear",
        align_corners=False,
    )

    #print(masks.unsqueeze(0))

    # loss = F.binary_cross_entropy_with_logits(masks.unsqueeze(0), torch.zeros_like(masks.unsqueeze(0)))
    # print("Negative Loss:", loss.item())

    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

def load_model(args):
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint_SAM).to(args.device).eval()
    net_high = MaskDecoderHigh(args.model_type).to(args.device).eval()
    net_low = MaskDecoderLow(args.model_type).to(args.device).eval()

    net_ckpt_high = args.checkpoint_net_high
    net_ckpt_low = args.checkpoint_net_low

    # Load High Res Checkpoint
    ckpt_h = torch.load(net_ckpt_high, map_location='cpu')
    if 'model' in ckpt_h:
        net_high.load_state_dict(ckpt_h['model'])
    else:
        net_high.load_state_dict(ckpt_h) # Direct load for your trained weights

    # Load Low Res Checkpoint
    ckpt_l = torch.load(net_ckpt_low, map_location='cpu')
    if 'model' in ckpt_l:
        net_low.load_state_dict(ckpt_l['model'])
    else:
        net_low.load_state_dict(ckpt_l) # Direct load for your trained weights

    return sam_model, net_high, net_low



def run_inference(high_res_image: torch.Tensor, low_res_image: torch.Tensor, box: list, args, test = False, sam_model=None, net_high=None, net_low=None):
    """
    Runs inference on the provided high and low-resolution images with a bounding box.

    Args:
        high_res_image (torch.Tensor): High-resolution image tensor (C, H, W)
        low_res_image (torch.Tensor): Low-resolution image tensor (C, H, W)
        box (list): Bounding box [x_min, y_min, x_max, y_max]
        args: Additional arguments for model loading.
    """
    # Load the SAM model and custom networks
    if test == False:
        sam_model, net_high, net_low = load_model(args)

    if sam_model is None or net_high is None or net_low is None:
        raise ValueError("Models must be provided for inference.")

    # Convert bounding box to tensor
    box_tensor = torch.tensor(box, device=args.device).unsqueeze(0).to(args.device)

    # Prepare input for high-res and low-res images
    dict_input_high = {
        'image': high_res_image.to(args.device),
        'boxes': box_tensor   # [x_min, y_min, x_max, y_max]
    }

    dict_input_low = {
        'image': low_res_image.to(args.device),
        #'boxes': box_tensor // 2 + 256 # Scale down the bounding box for low-res image
        'boxes': (box_tensor.float() * 0.5) + 256
    }

    # FOR DEBUGGING: Save the input images with boxes drawn
    import cv2
    
    def save_debug_input(img_tensor, box_tensor, filename):
        # 1. Convert Tensor (C,H,W) -> Numpy (H,W,C)
        img_vis = img_tensor.cpu().permute(1, 2, 0).numpy().copy()
        
        # 2. Draw the Box the model will see
        # box_tensor is likely [[x1, y1, x2, y2]]
        b = box_tensor.cpu().numpy()[0].astype(int)
        cv2.rectangle(img_vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2) # Green Box
        
        # 3. Save
        # Convert RGB to BGR for OpenCV saving
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_vis)
        print(f"Saved debug check to: {filename}")

    if test == False:
        # Check High Res Input
        save_debug_input(dict_input_high['image'], dict_input_high['boxes'], "debug_high_input.png")
        
        # Check Low Res Input
        save_debug_input(dict_input_low['image'], dict_input_low['boxes'], "debug_low_input.png")
    

    # Run inference using the SAM model
    with torch.no_grad():
        batched_output_high, interm_embeddings_high = sam_model([dict_input_high], multimask_output=False)
        batched_output_low, interm_embeddings_low = sam_model([dict_input_low], multimask_output=False)

        batch_len_high = len(batched_output_high)
        encoder_embedding_high = torch.cat([batched_output_high[i_l]['encoder_embedding'] for i_l in range(batch_len_high)], dim=0)
        image_pe_high = [batched_output_high[i_l]['image_pe'] for i_l in range(batch_len_high)]
        sparse_embeddings_high = [batched_output_high[i_l]['sparse_embeddings'] for i_l in range(batch_len_high)]
        dense_embeddings_high = [batched_output_high[i_l]['dense_embeddings'] for i_l in range(batch_len_high)]

        batch_len_low = len(batched_output_low)
        encoder_embedding_low = torch.cat([batched_output_low[i_l]['encoder_embedding'] for i_l in range(batch_len_low)], dim=0)
        image_pe_low = [batched_output_low[i_l]['image_pe'] for i_l in range(batch_len_low)]
        sparse_embeddings_low = [batched_output_low[i_l]['sparse_embeddings'] for i_l in range(batch_len_low)]
        dense_embeddings_low = [batched_output_low[i_l]['dense_embeddings'] for i_l in range(batch_len_low)]

        masks_sam_high, _, masks_wsi_token_high, wsi_image_embeddings_high = net_high(
            image_embeddings=encoder_embedding_high,
            image_pe=image_pe_high,
            sparse_prompt_embeddings=sparse_embeddings_high,
            dense_prompt_embeddings=dense_embeddings_high,
            multimask_output=False,
            wsi_token_only=False,
            interm_embeddings=interm_embeddings_high,
        )
        masks_sam_low, _, masks_wsi_token_low, wsi_image_embeddings_low = net_low(
            image_embeddings=encoder_embedding_low,
            image_pe=image_pe_low,
            sparse_prompt_embeddings=sparse_embeddings_low,
            dense_prompt_embeddings=dense_embeddings_low,
            multimask_output=False,
            wsi_token_only=False,
            interm_embeddings=interm_embeddings_low,
        )

        masks_wsi_token = masks_wsi_token_high + masks_wsi_token_low  
        masks_wsi_high = (masks_wsi_token @ wsi_image_embeddings_high).view(1, -1, 256, 256)
        masks_wsi_low = (masks_wsi_token @ wsi_image_embeddings_low).view(1, -1, 256, 256)


        masks_wsi_high = masks_wsi_high + masks_sam_high
        masks_wsi_high = postprocess_masks(
                masks_wsi_high,
                input_size=high_res_image.shape[-2:],
                original_size=high_res_image.shape[-2:],
            )

        mask_wsi_low = masks_wsi_low + masks_sam_low
        masks_wsi_low = postprocess_masks(
                masks_wsi_low,
                input_size=low_res_image.shape[-2:],
                original_size=low_res_image.shape[-2:],
            )

    # Return final mask prediction
    return masks_wsi_high, masks_wsi_low

def start_infer(test = False, args = None):
    # Parse arguments from the command line
    if test is False:
        args = parse_args()

    # Parse bounding box input
    bounding_box = args.bounding_box
    #bounding_box_low = torch.tensor(bounding_box) // 2 + 256
    bounding_box_low = (torch.tensor(bounding_box).float() * 0.5) + 256

    # Load high-resolution and low-resolution images from the given paths
    high_res_image = load_image(args.high_res_image)  # (C, H, W)
    low_res_image = load_image(args.low_res_image)  # (C, H, W)

    # Run inference
    mask_prediction_high, mask_prediction_low = run_inference(high_res_image, low_res_image, bounding_box, args, test)

    #print(min(mask_prediction.flatten().cpu().numpy()), max(mask_prediction.flatten().cpu().numpy()))
    mask_prob_high = torch.sigmoid(mask_prediction_high)
    mask_prob_low = torch.sigmoid(mask_prediction_low)
    
    # 2. Threshold: If prob > 0.5, it's cancer (1.0). Otherwise background (0.0).
    mask_prediction = (mask_prob_high > 0.5).float()
    mask_prediction_low = (mask_prob_low > 0.5).float()

    if test == False and args.visual == True:
        fig, ax = plt.subplots(1, 4, figsize=(25, 25))
        ax[0].imshow(high_res_image.permute(1,2,0).cpu().numpy())
        show_box(bounding_box, ax[0])
        ax[0].set_title("Input High-res Image with Bounding Box")
        ax[1].imshow(low_res_image.permute(1,2,0).cpu().numpy())
        show_box(bounding_box_low, ax[1])
        ax[1].set_title("Input Low-res Image with Bounding Box")
        ax[2].imshow(high_res_image.permute(1,2,0).cpu().numpy())
        show_mask(mask_prediction[0].cpu().numpy(), ax[2])
        show_box(bounding_box, ax[2])
        ax[2].set_title("High-res Image with Prediction")

        ax[3].imshow(low_res_image.permute(1,2,0).cpu().numpy())
        show_mask(mask_prediction_low[0].cpu().numpy(), ax[3])
        show_box(bounding_box_low, ax[3])
        ax[3].set_title("Low-res Image with Prediction")
        # plt.show()
        # plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig('examples/high_pred.png', bbox_inches="tight")

    return mask_prediction, mask_prediction_low

if __name__ == "__main__":
    start_infer(test = False)
    print('Finished!')
