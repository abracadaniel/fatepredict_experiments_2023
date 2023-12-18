# FatePredict Experiments
Various experiments for Cell Instance Segmentation and Tracking in 3D microscopy timelapse images.

## Installation
Many of the libraries has deprecated depedencies in newer versions, so use the specified versions to avoid dependency incompatibilities. 
I have setup multiple environments using anaconda, as some of the libraries I have been working with would cause dependency incompatibilities with eachother.
I use anaconda to manage the python environments.

### Basic Environment
1. `conda create --name fatepredict python=3.8`
2. `conda activate fatepredict`
3. `pip install -r requirements.txt`

### Linajea Environment
This was a bit tricky to get working on a M2 MacBook, as some of the packages are not built for this. I had to make some modifications in the pylp library to get it to install, which is why the pylp package is being installed from a local folder.

1. `conda create --name fatepredict_linajea python=3.10 cython boost pytorch pyscipopt -c pytorch -c funkey -c conda-forge`
2. `conda activate fatepredict_linajea`
3. Install Gurobi solver @ https://www.gurobi.com/
4. `pip install Repos/pylp`
5. `pip install -r requirements_linajea.txt`

### MultiPlanar U-Net Environment
1. `conda create --name fatepredict_mpunet python=3.9.18`
2. `conda activate fatepredict_mpunet`
3. `pip install -r requirements_mpunet.txt`
4. `git clone https://github.com/perslev/MultiPlanarUNet.git`
5. Change the tensorflow version in MultiPlanarUNet/requirements.txt to 2.5.0
6. `pip install MultiPlanarUNet`


## Content
Here I will explain some of the contents in this project folder.

### Linajea directory
Source code for solving instance segmentation and tracking using the Linajea package.

### 0_Data.ipynb
Notebook for looking at the dataset

### 1_WaterZ_Segmentation.ipynb
Experiments using Watershed and WaterZ for instance segmentation.
Uses the fatepredict conda environment.

### 2_StarDist_Segmentation.ipynb
Experiments using StarDist for instance segmentation.
Uses the fatepredict conda environment.

### 3_Basic_tracking.ipynb
Experiments using various tracking algorithms on instance segmentation results.
Uses the fatepredict conda environment.

### 4_2D_tracking_using_StarDist.ipynb
Experiments using StarDist in XYT for generating tracks.
Uses the fatepredict conda environment.

### 5_3D_MultiPlane_tracking.ipynb
Notebook for connecting the 2D tracks from #4 with the 3D instance segmented fragments. Incomplete.
Uses the fatepredict conda environment.

### 6_Linajea.ipynb
Experiments using the Linajea method for tracking.
Uses the fatepredict_linajea conda environment.

### 7_CompareHOTA.ipynb
Comparison of all the tracking methods using the HOTA measure.
Uses the fatepredict conda environment.

### 8_MultiPlanar.ipynb
Notebook for preparing the data to use with the MultiPlanar U-Net, for creating a binary mask of the cell fragments.
Training and prediction using the U-Net was done using the `mp` command, specified in the [MultiPlanar U-Net GitHub Repo](https://github.com/perslev/MultiPlanarUNet).
Uses the fatepredict_mpunet conda environment.

### trackers.py
File containing various helper functions, performance measures and tracking algorithms.
Tracking Algorithms:
- track_segmented: Tracking using IoU and own algorithm for assignment.
- track_segmented_hung_cent: Tracking using centroid distance and Hungarian algorithm for assignment.
- track_segmented_hung_earthmover: Tracking using wasserstein (Earthmover) distance and Hungarian Algorithm for assignment.
- track_segmented_hung: Tracking using IoU and Hungarian algorithm for assignment.

### trackeval_fatepredict.py
Wrapper to enable the use of the use of the [TrackEval Package](https://github.com/JonathonLuiten/TrackEval), implemeting the HOTA measure, with the FatePredict data format (TZYX).

## Data
The data used for these experiments is not distributed here. To gain access to the data, please contact Silja Heilmann @ silja.heilmann@sund.ku.dk.