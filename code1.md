### CosmoBench Dataset Tutorial
This notebook gives an overview of the datasets and tasks. Specifically:
- Point cloud datasets: CAMELS (1k clouds, 588-4511 points per cloud)
  - Task 1: Predicting cosmological parameters from point positions
  - Task 2: Predicting velocities from positions
- Merger tree datasets: CS-Trees (25k trees, 121-37865 nodes per tree); infilling-Trees (coarsened trees from the 200 largest CS-Trees)
  - Task 3: Predicing cosmological parameters from merger trees
  - Task 4: Reconstructing fine-grained trees from coarsened trees


```python
import os 
import subprocess
import torch 
import numpy as np
import math
import argparse
import pathlib
import h5py
import time
import matplotlib.pyplot as plt
import pickle
import json
from itertools import product

from models.cloud_param.simple_param import load_position_h5
from models.cloud_velocity.simple_velocity import load_point_cloud_h5

import warnings
warnings.filterwarnings('ignore')
```

### Point cloud datasets
- Quijote download url: https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/Quijote/
- CAMELS-SAM download url: https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/galaxies/
- CAMELS download url: https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS/
- Store the downloaded data in `DATA_DIR`. 

For demo purpose, we will download CAMELS (the smallest point cloud dataset) on-the-fly, exploring the point clouds as well as running some simple baselines

#### Data exploration


```python
data_dir = "CosmoBench_CAMELS"
os.makedirs(data_dir, exist_ok=True)

url = "https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS/"
pattern = "ALL_galaxies_*.hdf5" #only download the point cloud data (excluding the precomputed 2PCF)

cmd = [
    "wget", "-r", "-l1", "-nd", "-A", pattern,
    "-P", data_dir, url
]
subprocess.run(cmd)
```

After downloading, `DATA_DIR` should contain the train/val/test splits of CAMELS point clouds, in the form `ALL_galaxies_*.hdf5` 

We can read the point clouds from the H5 files as follows


```python
#specify the h5 file dataset group key
data_dir = "CosmoBench_CAMELS"

if "Quijote" in data_dir:
    grp_key = "BSQ"
elif "CAMELS" in data_dir:
    grp_key = "LH"
#train/val/test files
h5_train = f"{data_dir}/ALL_galaxies_train.hdf5"
h5_val = f"{data_dir}/ALL_galaxies_val.hdf5"
h5_test = f"{data_dir}/ALL_galaxies_test.hdf5"

idx = 0
x,y,z = load_position_h5(h5_train, idx=idx, data_name=grp_key, device="cpu")
x,y,z,vx,vy,vz = load_point_cloud_h5(h5_train, idx=idx, data_name=grp_key, device='cpu')
```


```python
fig = plt.figure(figsize=(10,10), dpi=200)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(x,y,z, color='tab:blue', s=0.5, alpha=0.5)
ax2.scatter(x,y,z, color='tab:blue', s=0.5, alpha=0.5)
ax2.quiver(x,y,z,vx,vy,vz, length=0.5, normalize=True, color='tab:purple', linewidth=0.5)

ax1.set_title('example point cloud (position only)')
ax2.set_title('example point cloud (with velocity)')

plt.show()
```

