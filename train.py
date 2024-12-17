import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
from model import UNET
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    draw_plot,
    load_data
)

#Parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "frames/"
TRAIN_MASK_DIR = "mask/"
VAL_IMG_DIR = "val_frames/"
VAL_MASK_DIR = "val_mask/"

def train_fn(loader, model, optimizer, loss_fn, scaler, step_loss):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        #targets = targets.to(device=DEVICE)
        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backword
        
        optimizer.zero_grad()
        output = model(data).double()
        #loss = nn.CrossEntropyLoss()(output, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step_loss.append(loss.item())
        

        loop.set_postfix(loss=loss.item())

def main():
    trainingEpoch_loss = []
    validationEpoch_loss = []
    dice_score = []

    trainingEpoch_loss, validationEpoch_loss, dice_score = load_data()

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width =IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transformers = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ]
    )

    #model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    model = smp.Unet(
        encoder_name="tu-efficientnetv2_m",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 # timm-efficientnet-b2	
        encoder_weights=None,#"imagenet",     # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )

    model = model.to(DEVICE)
   
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transformers
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model,loss_fn=loss_fn, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        step_loss = []




        train_fn(train_loader, model, optimizer, loss_fn, scaler, step_loss)
        trainingEpoch_loss.append(np.array(step_loss).mean())

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict,
        }
        if epoch % 20 == 0:
            save_checkpoint(checkpoint, "my_checkpoint_" + str(epoch) + ".pth.tar" )

        save_checkpoint(checkpoint)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device =DEVICE
        )

        d_score = check_accuracy(val_loader, model, loss_fn=loss_fn, device=DEVICE, validationEpoch_loss=validationEpoch_loss)
        dice_score.append(d_score.cpu())
        draw_plot(validation_loss=validationEpoch_loss, training_loss=trainingEpoch_loss, avg_dice=dice_score)



if __name__ == "__main__":
    main()