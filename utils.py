import torch
import torchvision
from dataset import DataSet
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import json

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> SAVING CHECKPOINT")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> LOADING CHECKPOINT")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = DataSet(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,shuffle=True,)
    val_ds = DataSet(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, loss_fn, device="cuda", validationEpoch_loss=[]):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_dice_score = 0
    num_of_scores = 0
    model.eval()
    validation_loss = []
    with torch.no_grad():
        for x,y in loader:
            
            x = x.to(device)

            targets = y.float().unsqueeze(1).to(device=device)
            with torch.cuda.amp.autocast():
                predictions = model(x)
                loss = loss_fn(predictions, targets)

            validation_loss.append(loss.item())

            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2* (preds * y).sum())/ ((preds + y).sum() + 1e-8)

            print(
                f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
            )
            print(f"Dice score: {dice_score/len(loader)}")
            total_dice_score += dice_score/len(loader)
            num_of_scores += 1

    validationEpoch_loss.append(np.array(validation_loss).mean())
    print(f"Average Dice score: {total_dice_score/num_of_scores}")
    model.train()
    return (total_dice_score/num_of_scores)

    


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/true_{idx}.png")

    model.train()

def draw_plot(
  training_loss, validation_loss, avg_dice      
):
    print("=> DRAWING PLOT")
    np.savetxt("training_loss.csv", np.array(training_loss), delimiter=',')
    np.savetxt("validation_loss.csv", np.array(validation_loss), delimiter=',')
    np.savetxt("avg_dice.csv", np.array(avg_dice), delimiter=',')


    plt.plot(training_loss, label='train_loss')
    plt.plot(validation_loss,label='val_loss')
    plt.plot(avg_dice, label='dice_score')
    #plt.legend()
    plt.savefig('loss.png')


def load_data():
    training_loss = np.loadtxt("training_loss.csv",delimiter=',').tolist()
    validation_loss = np.loadtxt("validation_loss.csv",delimiter=',').tolist()
    avg_dice = np.loadtxt("avg_dice.csv",delimiter=',').tolist()

    return (training_loss, validation_loss, avg_dice)