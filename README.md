# U-Net for TPVM

## Train a model
```
python train.py
```

## Create Dataset
Before running train.py make sure to fill following folder withe the training and annotation data. 
-   ./frames - Frames to train on 
-   ./mask - A black and white mask that highlights the marker
-   ./val_frames - same as above
-   ./val_mask - same as above

Functions from ./helper-scripts can be used to create an artificial dataset. 

## Results
Checkpoint gets saved every iteration. Every 20 iteration an additional backup gets saved.
The folder ./saved_images shows the results of the prediction as image. 