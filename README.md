
# mazegan

Solving maze via image-to-image translation.

## Maze Generator

Collected 5000 pair maze images (solved and unsolved).  
The maze is generated randomly by recursive backtracking algorithm.  
See the repository bellow for more details.  
[keesiemeijer/maze-generator](https://github.com/keesiemeijer/maze-generator)

## Model

The model is mostly based on pix2pix.  

### Generator

U2Net  
[paper](https://arxiv.org/abs/2005.09007) | [code](https://github.com/NathanUA/U-2-Net)  
A model for semantic segmentation.  
Replaced with UNet Generator in pix2pix.  
Modified to use spectral normalization on every convolution layers.  

### Discriminator

PatchGAN with receptive field of 70x70 used in pix2pix.  
[paper](https://arxiv.org/abs/1611.07004) | [code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
Modified to use spectral normlization on every convolution layers.  

## Training

- Loss

    Non-saturated loss + 100 * L1 loss

- Optimizer

    Adam(lr=0.0002, betas=(0.5, 0.999))

- No learning rate schedule

- Batch size

    16

- Max iterations (epochs)

    31000 (100)

- Used AMP (fp16)

- Training time

    About 8h30m

## Author

[STomoya](https://github.com/STomoya)
