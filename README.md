Non-invasive ultrasound based pacemaker system
------------------------------------------------------------------------------

## Models

The model used is [EchoNet-Dynamic](https://github.com/echonet/dynamic?tab=readme-ov-file), which consists of its two main components:

- Semantic segmentation of the left ventricle
- Prediction of ejection fraction

For more details, see the accompanying paper,

> [**Video-based AI for beat-to-beat assessment of cardiac function**](https://www.nature.com/articles/s41586-020-2145-8)<br/>
> David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich, Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. <b>Nature</b>, March 25, 2020. https://doi.org/10.1038/s41586-020-2145-8

### Installation

Dependencies:

  - NumPy
  - PyTorch
  - Torchvision
  - OpenCV
  - skimage
  - sklearn
  - tqdm

For installation, clone the repository and run

```zsh
pip install .
```

By default, the data is saved in `a4c-video-dir/` directory.

This path can be changed by creating a configuration file named `echonet.cfg` (an example configuration file is `example.cfg`).

### Part1 - Semantic Segmentation of the left ventricle

Run the following command:

```zsh
echonet segmentation --save_video
```

The output results will be located in `output/segmentation/deeplabv3_resnet50_random/` and will include the following components:

1. `log.csv`: training and validation losses
2. `best.pt`: checkpoint of weights for the model with the lowest validation loss
3. `size.csv`: the estimated size of the left ventricle in each frame and the indicator for the beginning of a beat
4. `videos`: segmented video

After training, you can run the following command to segment your test dataset separately. However, you need to place your dataset into a folder in advance and register it in the `FileList.csv` and `VolumeTracings.csv` files.

```zsh
echonet segmentation --save_video --run_test --weights YOUR_PATH_TO_best.pt
```

### Part2 - Prediction of ejection fraction

Run the following command:

```zsh
echonet video
```

The output results will be located in `output/video/r2plus1d_18_32_2_pretrained/`，and will include the following components:

1. `log.csv`: training and validation losses
2. `best.pt`: checkpoint of weights for the model with the lowest validation loss
3. `test_predictions.csv`: ejection fraction prediction for subsampled clips

After training, you can run the following command to predict your test dataset:

```zsh
echonet video --run_test --weights YOUR_PATH_TO_best.pt
```

## Server

To run the server, run `node server.js`. The server will be running on port 8080.
