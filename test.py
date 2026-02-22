import torch
import torch.nn.functional as F
import sys, os
import warnings
import matplotlib.pyplot as plt
import logging

from model.initial_sapmedsam import init_network
from trainer import train_net, get_data

from model.initial_sapmedsam import SAPMedSAM

from monai.metrics import compute_hausdorff_distance, DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai.losses import DiceCELoss, DiceLoss
from tqdm import tqdm
from model.utils import AverageMeter

warnings.filterwarnings("ignore", category=UserWarning)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def test_net(model, test_loader, device, pretrained_weights, save_dir=None):

    model.eval()

    if os.path.exists(pretrained_weights):
        print(f"Loading checkpoint from {pretrained_weights}...")
        checkpoint = torch.load(pretrained_weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    criterion = DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True, softmax=False,
                           lambda_dice=1, lambda_ce=1)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean",get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean",get_not_nans=False)

    test_loss = AverageMeter("Val Loss", ":.4f")

    with torch.no_grad():

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {1} Validation", leave=True, position=0)

        for idx, batch_data in enumerate(test_loader_tqdm):
            inputs, labels, name = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["name"]

            with torch.cuda.amp.autocast():
                pred, _ = model(inputs)

                pred = F.interpolate(
                    pred,
                    size=(labels.shape[2], labels.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            dice_metric(y_pred=pred, y=labels.long())
            iou_metric(y_pred=pred, y=labels.long())
            hd95_metric(y_pred=pred, y=labels.long())

            save_path = os.path.join(save_dir, name[0] + f".png")
            plt.imsave(save_path, pred.squeeze().cpu().numpy(), cmap="gray")

        mean_dice = dice_metric.aggregate().item()
        mean_iou = iou_metric.aggregate().item()
        mean_hausdorff = hd95_metric.aggregate().item()

    return mean_dice, mean_iou, mean_hausdorff



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrained_weights = ""

    save_dir = ""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_loader, val_loader = get_data(base_folder="",
                                        batch_size=1, return_test=True, debug=False)

    image_encoder, prompt_encoder, mask_decoder, mask_decoder1 = init_network(device=device)

    model = SAPMedSAM(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        mask_decoder1=mask_decoder1,
    ).to(device)


    dice, iou, hd95 = test_net(model, test_loader, device, pretrained_weights, save_dir)

    print(f"average dice: {dice:.4f} average iou: {iou:.4f} average hd95: {hd95:.4f}")

