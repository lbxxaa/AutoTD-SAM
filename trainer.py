import os
import random
import pandas as pd
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset

from monai.metrics import compute_hausdorff_distance, DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.losses import DiceCELoss, DiceLoss
import torchvision.transforms as transforms
import torch
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn.functional as F
from torch.optim import AdamW
import yaml
from tqdm import tqdm
from pathlib import Path
from model.loss import box_giou_loss, box_l1_loss

def random_rot_flip_pil(image, label):
    k = random.randint(0, 3)
    image = image.rotate(90 * k)
    label = label.rotate(90 * k)

    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)

    return image, label

def random_rotate_pil(image, label):
    angle = random.randint(-20, 20)
    image = image.rotate(angle, resample=Image.BILINEAR)
    label = label.rotate(angle, resample=Image.NEAREST)
    return image, label

class Dataset(Dataset):
    def __init__(self, data, image_transform=None, label_transform=None, augment=False):

        self.data = data
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        label_path = self.data[idx]["label"]
        box_path = self.data[idx]["box"]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # ======================
        if self.augment:
            if random.random() < 0.5:
                image, label = random_rot_flip_pil(image, label)
            if random.random() < 0.5:
                image, label = random_rotate_pil(image, label)
        # ======================

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        img_h, img_w = image.shape[1], image.shape[2]

        with open(box_path, 'r') as f:
            line = f.readline().strip()
            values = line.split()
            values = list(map(float, values))

        x_center, y_center = values[1], values[2]
        box_width, box_height = values[3], values[4]

        x1 = int((x_center - box_width / 2) * img_w)
        y1 = int((y_center - box_height / 2) * img_h)
        x2 = int((x_center + box_width / 2) * img_w)
        y2 = int((y_center + box_height / 2) * img_h)

        box = torch.tensor([x1, y1, x2, y2], dtype=torch.float)

        return {"image": image, "label": label, "box": box, "name": Path(image_path).stem}

def get_data(base_folder=None,
             batch_size=1, return_test=False, debug=False):
    # Load image names for training, validation, and testing
    with open(os.path.join(base_folder, 'list', 'train.yaml'), 'r') as f:
        img_name_list_train = yaml.load(f, Loader=yaml.BaseLoader)

    with open(os.path.join(base_folder, 'list', 'val.yaml'), 'r') as f:
        img_name_list_val = yaml.load(f, Loader=yaml.BaseLoader)

    with open(os.path.join(base_folder, 'list', 'test.yaml'), 'r') as f:
        img_name_list_test = yaml.load(f, Loader=yaml.BaseLoader)

    # Prepare file paths for training, validation, and testing
    train_files = [{"image": f'{base_folder}/train/input/{name}.jpg', "label": f'{base_folder}/train/label/{name}.png', "box": f'{base_folder}/train/box/{name}.txt'} for name in img_name_list_train]
    val_files = [{"image": f'{base_folder}/val/input/{name}.jpg', "label": f'{base_folder}/val/label/{name}.png', "box": f'{base_folder}/val/box/{name}.txt'} for name in img_name_list_val]
    test_files = [{"image": f'{base_folder}/test/input/{name}.jpg', "label": f'{base_folder}/test/label/{name}.png', "box": f'{base_folder}/test/box/{name}.txt'} for name in img_name_list_test]

    if debug:
        train_files = train_files[:int(len(train_files) * 0.1)]
        val_files = val_files[:int(len(val_files) * 0.1)]
        test_files = test_files[:int(len(test_files) * 0.1)]
        print(f'10% of data is used for training, validation, and testing. (debugging)')
    else:
        train_files = train_files[:int(len(train_files) * 1)]
        val_files = val_files[:int(len(val_files) * 1)]
        test_files = test_files[:int(len(test_files) * 1)]

    print('Training files:', len(train_files), '\nValidation files:', len(val_files), '\nTest files:', len(test_files))

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    label_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # If test data is needed, return the test loader
    if return_test:
        test_ds = Dataset(data=test_files, image_transform=image_transforms, label_transform=label_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        val_ds = Dataset(data=val_files, image_transform=image_transforms, label_transform=label_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return test_loader, val_loader
    else:
        # Return train and validation loaders
        train_ds = Dataset(data=train_files, image_transform=image_transforms, label_transform=label_transforms, augment=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_ds = Dataset(data=val_files, image_transform=image_transforms, label_transform=label_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


def validate_net(model, val_loader, device, logger=None, epoch=0):
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=False)

    with torch.no_grad():

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=True, position=0)

        for idx, batch_data in enumerate(val_loader_tqdm):
            inputs, labels, boxs = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["box"].to(device)

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

        mean_dice = dice_metric.aggregate().item()
        mean_iou = iou_metric.aggregate().item()
        mean_hausdorff = hd95_metric.aggregate().item()

        if logger:
            logger.info(f"Epoch {epoch + 1} , Dice Score: {mean_dice:.6f} , IoU: {mean_iou:.6f} , Hausdorff: {mean_hausdorff:.6f}")

    return mean_dice, mean_iou, mean_hausdorff


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

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

def train_net(model, train_loader, val_loader, device, args, logger=None, writer=None):

    max_epochs = args.max_epochs
    val_interval = args.val_interval

    ce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    optimizer = AdamW([i for i in model.parameters() if i.requires_grad], lr=args.base_lr, weight_decay=1e-7)
    optimizer_box = AdamW(model.boxpred.parameters(), lr=args.base_lr, weight_decay=1e-7)

    best_dice = 0.0

    if os.path.exists(args.pretrained_weights):
        logger.info(f"Loading checkpoint from {args.pretrained_weights}...")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    for epoch in range(max_epochs):

        model.train()

        scaler = torch.cuda.amp.GradScaler()

        epoch_mask_loss = 0
        epoch_box_loss = 0
        epoch_slice = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", total=len(train_loader))

        for idx, batch_data in enumerate(train_loader_tqdm):
            inputs, labels, boxs = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["box"].to(device)

            scale = torch.tensor([inputs.shape[2], inputs.shape[3], inputs.shape[2], inputs.shape[3]], device=inputs.device)

            with torch.cuda.amp.autocast():
                pred, pred_boxs = model(inputs)

                pred = F.interpolate(
                    pred,
                    size=(labels.shape[2], labels.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

                lc1, lc2 = 0.6, 0.4
                loss_ce = ce_loss(pred, labels)
                loss_dice = dice_loss(pred, labels.float())
                mask_loss = lc1 * loss_ce + lc2 * loss_dice

                # ==== Box Loss ====
                pred_boxs = box_to_xyxy(pred_boxs, inputs.shape[2], inputs.shape[3])
                pred_boxs = pred_boxs.squeeze(1)
                pred_boxs = pred_boxs / scale
                boxs = boxs / scale
                box_loss = 0.7 * box_giou_loss(pred_boxs, boxs) + 0.3 * box_l1_loss(pred_boxs, boxs)

            scaler.scale(mask_loss).backward()
            scaler.step(optimizer)

            scaler.scale(box_loss).backward()
            scaler.step(optimizer_box)

            scaler.update()
            optimizer.zero_grad()
            optimizer_box.zero_grad()

            epoch_mask_loss += mask_loss.item()
            epoch_box_loss += box_loss.item()
            epoch_slice += 1
            train_loader_tqdm.set_postfix(mask_loss=mask_loss.item(), box_loss=box_loss.item())

        avg_mask_loss = epoch_mask_loss / epoch_slice
        avg_box_loss = epoch_box_loss / epoch_slice

        logger.info(f"epoch {epoch + 1} average loss: {avg_mask_loss:.7f}")
        writer.add_scalar('train_mask_loss', avg_mask_loss, epoch + 1)
        writer.add_scalar('train_box_loss', avg_box_loss, epoch + 1)

        ################################################################################################################

        if (epoch + 1) % val_interval == 0:
            mean_dice, mean_iou, mean_hd95 = validate_net(model, val_loader, device, logger, epoch)

            csv_path = os.path.join(args.log_path, "train.csv")

            epoch_data = {
                "Epoch": epoch + 1,
                "Train Loss": f"{avg_mask_loss:.6f}",
                "Dice Score": f"{mean_dice:.6f}",
                "IoU Score": f"{mean_iou:.6f}",
                "HD95 Score": f"{mean_hd95:.6f}"
            }

            df = pd.DataFrame([epoch_data])

            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)

            if mean_dice > best_dice:
                best_dice = mean_dice

                best_checkpoint_path = os.path.join(args.log_path, "best.pth.tar")

                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)

                save_checkpoint(model, optimizer, best_checkpoint_path)

                if logger:
                    logger.info(f"New best model saved with Dice Score: {best_dice:.6f}")
