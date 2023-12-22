# Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis
### [Project Page](https://dynamic3dgaussians.github.io/) | [Paper](https://arxiv.org/pdf/2308.09713.pdf) | [ArXiv](https://arxiv.org/abs/2308.09713) | [Tweet Thread](https://twitter.com/JonathonLuiten/status/1692346451668636100) | [Data](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip) | [Pretrained Models](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip)
Official implementation of our approach for modelling the dynamic 3D world as a set of 3D Gaussians that move & rotate over time. This extends Gaussian Splatting to dynamic scenes, with accurate novel-view synthesis and dense 3D 6-DOF tracking.<br><br>
[Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis](https://dynamic3dgaussians.github.io/)  
 [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/) <sup>1, 2</sup>,
 [Georgios Kopanas](https://grgkopanas.github.io/) <sup>3</sup>,
 [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/) <sup>2</sup>,
 [Deva Ramanan](https://www.cs.cmu.edu/~deva/) <sup>1</sup> <br>
 <sup>1</sup> Carnegie Mellon University, <sup>2</sup> RWTH Aachen University, <sup>3</sup> Inria & Universite Cote d’Azur, France <br>
 International Conference on 3D Vision (3DV), 2024 <br>
jonoluiten@gmail.com

<p float="middle">
  <img src="./teaser_figure.png" width="99%" />
</p>

## Installation
```bash
# Install this repo (pytorch)
git clone git@github.com:JonathonLuiten/Dynamic3DGaussians.git
conda env create --file environment.yml
conda activate dynamic_gaussians

# Install rendering code (cuda)
git clone git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .
```

## Run visualizer on pretrained models
```bash
cd Dynamic3DGaussians
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip  # Download pretrained models
unzip output.zip
python visualize.py  # See code for visualization options
```

## Train models yourself
```bash
cd Dynamic3DGaussians
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip  # Download training data
unzip data.zip
python train.py 
```

## Code Structure
I tried really hard to make this code really clean and useful for building upon. In my opinion it is now much nicer than the original code it was built upon.
Everything is relatively 'functional' and I tried to remove redundant classes and modules wherever possible. 
Almost all of the code is in [train.py](./train.py) in a few core functions, with the overall training loop clearly laid out.
There are only a few other helper functions used, divided between [helpers.py](helpers.py) and [external.py](external.py) (depending on license).
I have split all useful variables into two dicts: 'params' (those updated with gradient descent), and 'variables' (those not updated by gradient descent).
There is also a custom visualization codebase build using Open3D (used for the cool visuals on the website) that is entirely in [visualize.py](visualize.py).
Please let me know if there is anyway you think the code could be cleaner. 


## Camera Coordinate System
This code uses uses the OpenCV camera co-ordinate system (same as COLMAP). This is different to the blender / standard NeRF camera coordinate system. The conversion code between the two can be found [here](https://github.com/NVlabs/instant-ngp/blob/9f6ab886306ce1f9b359d3856e8a1907ce8b8c8b/scripts/colmap2nerf.py#L350)


## Differences to paper
This codebase contains some significant changes from the results presented in the currently public version of the paper.
Both this codebase and the corresponding [paper](https://arxiv.org/pdf/2308.09713.pdf) are work-in-progress and likely to change in the near future.
Until I find time to update the paper (ETA Feb 4th) the code here is the more up-to-date public facing version of these two.

Differences:
 - In the paper we 'hard fixed' the colour to be perfectly consistent over time for each Gaussian by simply not updating it after the first timestep.
In this codebase the colour is only 'soft fixed'. e.g. it is updated over time, but there is a regularization loss added which 'soft enforces' it to be consistent over time.
 - Because we know the ground-plane of the scenes we are using a 'floor loss' to enforce Gassians don't go below the floor.

Please let me know if there are any other differences between the paper and the code so that I can include them here and remember to include them in future version of the paper.


## Partial code release
So far we have released two parts of the code: training and visualization.
There are three further parts to be released in the future when I find time to clean them up (ETA Feb 4th):
 - Evaluation code for evaluating the method for both novel-view-synthesis and tracking.
 - Data preparation code, to create the cleaned dataset (which I have provided), from the raw CMU panoptic capture data.
 - Code for creative editing of scenes (scene composition, etc).

Previously I hoped to have this done in December, unfortunately I didn't manage this and I am now aiming for Feb 4.

## Calls for contributions: Let's make this code better together!
Happy to work together to make this code better. If you want to contrib either open and issue / pull request, or send me an email.

### Speeding up the code
I do a number of dumb things which slows the code down ALOT. If someone is motivated improving these could significantly speed up training time.
 - To do the 'fg/bg' segmentation loss I am doing the full rendering process twice. By diving into the cuda a little, changing this could easily make training ~2x faster.
However, note that for full reproducibility maybe this should only be done after the first timestep, as in the first timestep the gradients from just the colour rendering are used for densification.
 - The current implementation of the local-rigidity loss in pytorch slows the code down MUCH more than it should. Currently it adds something like 20ms per training iter (e.g. 50 iter/s with, 100 without)
however I have a compiled jax version that is much faster only adding ~1ms which wouldn't really slow down training at all). Not sure how to speed this up in pure pytorch but there might be away. 
Other ideas include upgrading to pytorch 2.0 and using compile, writing this part in cuda, or somehow using jax here (or everywhere).
 - Lots of other parts of this pytorch code are not super efficient. Lots of room to make speedups.
 - Potentially the whole code could be ported to pure cuda. E.g. see [here](https://github.com/MrNeRF/gaussian-splatting-cuda).
 
### Visualization
In this codebase we provide an open3D based dynamic visualizer. This is makes adding 3D effects (like the track trajectories) really easy, although it definitely makes visualization slower than it should be. 
E.g. the code renders the scene at 800 FPS, but including open3D in order to display it on the scene (and add camera controls etc) slows it down to ~30 FPS.

I have seen lots of cool renderers for Gaussians for static scenes. It would be cool to make my dynamic scenes work on these.

In particular, I have seen various things that (a) somehow run on my phone and old laptop (e.g. [here](https://gsplat.tech/) and [here](https://huggingface.co/spaces/cakewalk/splat)) (b) run on VR headsets (e.g. [here](https://twitter.com/charshenton/status/1704358063036375548) and [here](https://twitter.com/s1ddok/status/1696249177250931086)) (c) run in commonly used tools like unity (e.g. [here](https://github.com/aras-p/UnityGaussianSplatting))

Dylan made a helpful list that can be found [here](https://huggingface.co/spaces/dylanebert/list-of-splats)

### Better (or no) FG / BG segmentations
The current FG/BG segmentations I use are REALLY bad. I made them very quickly by using simple background subtraction using a background image (image with no objects) for each camera with some smoothing. The badness of these segmentation masks causes a noticable degradation of the results. Especially around the feet of people / near the floor. It should be very easy to get much better segmentation masks (e.g. using pretrained networks), but I think it also probably isn't too hard to get the method to work without segmentation masks as all.

## Further research
There are ALOT of cool things still to be done building upon Dynamic 3D Gaussians. If you're doing so (especially research projects) feel free to reach out if you want to discuss (email / issue / twitter)

## Notes on license
The code in this repository (except in external.py) is licensed under the MIT licence.

However, for this code to run it uses the cuda rasterizer code from [here](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth),
as well as various code in [external.py](./external.py) which has been taken or adapted from [here](https://github.com/graphdeco-inria/gaussian-splatting).
These are required for this project, and for these a much more restrictive license from Inria applies which can be found [here](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).
This requires express permission (licensing agreements) from Inria for use in any commercial application, but is otherwise freely distributed for research and experimentation.


## Citation
```
@inproceedings{luiten2023dynamic,
  title={Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis},
  author={Luiten, Jonathon and Kopanas, Georgios and Leibe, Bastian and Ramanan, Deva},
  booktitle={3DV},
  year={2024}
}
```
