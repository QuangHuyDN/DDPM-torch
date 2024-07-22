# DDPM-torch
Implementation of the "Denoising Diffusion Probabilistic Models" (DDPM) using Pytorch, combined with Classifier-free guidance (CFG) and EMA for improved performance.

To run this repo, first install the dependencies using the ***requirements.txt*** (highly recommend to use environment manager like conda)
```
pip install -r requirements.txt
```

To start the training process, run
```
python conditional_ddpm.py \
--root_dir <root_dir> \
--run <run> \
--num_classes <num_classes> \
--size <size> \
--in_chans <in_chans> \
--batch_size <batch_size> \
--lr <lr> \
--epochs <epochs> \
--gpu_id <gpu_id>
```
During the training process, there are several settings can be added to the training process.
- --root_dir : directory to the training dataset (follow ImageNet structure), required.
- --run : name of that training run, use for checkpointing location, required.
- --num_classes : number of categories in the dataset.
- --size : size of the image, default 64.
- --in_chans : number of image color channels, default 3 (RGB).
- --batch_size : size of each data batch, default 16.
- --lr : learning rate, default 3e-4.
- --epochs : number of training epochs, default 300.
- --gpu_id : the id of the GPU used for training, default 0.
