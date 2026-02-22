import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import torch.nn as nn
from .utils import BoxPredictionHead
from .utils import LoRA_qkv

def box_to_xyxy(boxes, img_w, img_h):

    cx = boxes[..., 0] * img_w
    cy = boxes[..., 1] * img_h
    w  = boxes[..., 2] * img_w
    h  = boxes[..., 3] * img_h

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack([x1, y1, x2, y2], dim=-1)


def init_network(device=None, use_lora=True, lora_rank=4, lora_layers=None):
    sam = sam_model_registry["vit_b"](checkpoint="./model/ckpt/sam_vit_b_01ec64.pth")

    image_encoder = sam.image_encoder
    prompt_encoder = sam.prompt_encoder
    mask_decoder = sam.mask_decoder
    mask_decoder1 = sam.mask_decoder1

    image_encoder.to(device)
    prompt_encoder.to(device)
    mask_decoder.to(device)
    mask_decoder1.to(device)

# ------------------------------------------------------------------------------------------------
    if use_lora:
        if lora_layers is None:
            lora_layers = list(range(len(sam.image_encoder.blocks)))

        for p in sam.image_encoder.parameters():
            p.requires_grad = False

        for i, blk in enumerate(sam.image_encoder.blocks):
            if i in lora_layers:
                blk.attn.qkv = LoRA_qkv(blk.attn.qkv, r=lora_rank)

    image_encoder.pos_embed.requires_grad = True

    for name, param in image_encoder.named_parameters():
        if "rel_pos" in name:
            param.requires_grad = True
        if "conv_stem" in name:
            param.requires_grad = True

#------------------------------------------------------------------------------------------------

    for param in prompt_encoder.parameters():
        param.requires_grad = False

# ------------------------------------------------------------------------------------------------

    for param in mask_decoder.parameters():
        param.requires_grad = False

# ------------------------------------------------------------------------------------------------
#
    for param in mask_decoder1.parameters():
        param.requires_grad = True
#
# # ------------------------------------------------------------------------------------------------

    del sam

    return image_encoder, prompt_encoder, mask_decoder, mask_decoder1

class Model(nn.Module):
    def __init__(
            self,
            image_encoder,
            prompt_encoder,
            mask_decoder,
            mask_decoder1,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.mask_decoder1 = mask_decoder1

        self.boxpred = BoxPredictionHead(embed_dim=256, num_queries=1)

    def forward(self, image):

        B, _, H_img, W_img = image.shape

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        pred_box = self.boxpred(image_embedding.detach())

        boxs = box_to_xyxy(pred_box, W_img, H_img)

        # do not compute gradients for prompt encoder

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=boxs,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 1, 256) -> (B, 1, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        
        with torch.no_grad():
            sparse_embeddings1, dense_embeddings1 = self.prompt_encoder(
                points=None,
                boxes=boxs,
                masks=low_res_masks,
            )

        low_res_masks1, _ = self.mask_decoder1(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings1,  # (B, 1, 256) -> (B, 1, 256)
            dense_prompt_embeddings=dense_embeddings1,  # (B, 256, 64, 64)
            multimask_output=False,
        )


        return low_res_masks1, pred_box