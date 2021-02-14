


## Download Datasets
Check out the following [list of datasets](https://github.com/taozh2017/RGBD-SODsurvey#datasets)

* [NJU2K Dataset Download](https://paperswithcode.com/dataset/nju2k)

## Run

```
python main.py
```

## Code Layout:
```
.
├── blocks/                         #  Blocks of Paper
│   ├── compact_pyramid.py          # Compact Pyramid Refinement block
│   ├── depthwise_seperable.py      # Depthwise Seperable Conv. used in CPR
│   ├── implicit_depth.py           # Implicit Depth Restoration
│   ├── mobilenet.py                # MobileNetv2
│   ├── modality_fusion.py          # Cross-Modality Fusion 
│   └── rgb_attention.py            # RGB Attention - Eqn. (2)
│
├── checkpoints/                    # Saved models during training
│
├── data/                           # Datasets
│
├── main.py                         # Main driver 
│
├── model.py                        # Entire model implem.
│
├── README.md
│
└── utils/                          
    ├── dataset.py                  # pytorch Datasets
    ├── pytorch_msssim.py           # SSIM Loss implem.
    ├── solver.py                   # Trainer
    └── viz.py                      # Viz for validation

```
## References:

* [Pytorch MS-SSIM Implementation](https://github.com/VainF/pytorch-msssim)
* [Dice Loss Implementation](https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)


## Implementation Details

```
    MobileSal -  Implementation of paper as close as possible
    MobileSalTranspose -  Replaced upscale interpolation with transpose conv layers 
```

## Known Issues

Currently has training issues, Dice and BCE loss giving good scores to very bad outputs.
