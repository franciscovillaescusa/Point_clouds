#### Task 1: Predicting cosmological params from point positions
- Baseline: linear least squares on pairwise-distance (truncated at Rc) statistics


```python
from models.cloud_param.simple_param import Rc_dict_om, Rc_dict_s8, \
    compute_h5_features, load_param_h5, fit_least_squares, compute_R2, bootstrap_r2
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
prefix = "CAMELS-TNG" #['Quijote', 'CAMELS-SAM']
n_train, n_val, n_test, period = 600, 200, 200, 25 
Rc = Rc_dict_om[prefix]
```


```python
#this can take a few minutes...please be patient
X_train, _ = compute_h5_features(h5_train, grp_key, 
                                n_train, Rc, period, device)
X_test, _ = compute_h5_features(h5_test, grp_key, 
                                n_test, Rc, period, device)
```


```python
X_train = X_train.flatten(start_dim=0, end_dim=1)
X_train = X_train.to(device) #shape (4*12, n_cloud_train)

X_test = X_test.flatten(start_dim=0, end_dim=1) 
X_test = X_test.to(device)  #shape (4*12, n_cloud_test)

#load labels 
target_name = 'om' #'s8'
Y_train = load_param_h5(h5_train, target_name, device=device)
Y_test = load_param_h5(h5_test, target_name, device=device)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
```

    torch.Size([48, 600]) torch.Size([600]) torch.Size([48, 200]) torch.Size([200])



```python
w, b = fit_least_squares(X_train, Y_train)
#pred: clip into the range
lim = Rc_dict_om['limits'] #Rc_dict_s8['limits']
Y_pred_train = torch.clamp( w @ X_train + b , min=lim[0], max=lim[1])
Y_pred_test = torch.clamp( w @ X_test + b , min=lim[0], max=lim[1])

#Compute metrics
R2_train = compute_R2(Y_pred_train, Y_train)
R2_test = compute_R2(Y_pred_test, Y_test)
R2_boot, R2_boot_std = bootstrap_r2(Y_pred_test, Y_test )

print(f"{prefix} on {target_name}: R2_train={R2_train:.4f}, R2_test={R2_test:.4f}, R2_boot_std={R2_boot_std:.4f}")

```

    CAMELS-TNG on om: R2_train=0.8567, R2_test=0.7886, R2_boot_std=0.0269


#### Task 2: Predicting velocities from positions
- Baseline: linear least squares on powers of inverse distances


```python
from models.cloud_velocity.simple_velocity import compute_invPwrLaw_features
```


```python
#Feature Order and init matrices for linear least square fit
K, P = 10, 3
A = torch.zeros((K*P, K*P)).to(device)
b = torch.zeros((K*P,1)).to(device)
```


```python
idxTrain = range(0, n_train)
idxTest = range(0, n_test)

for t in idxTrain:
    x,y,z,vx,vy,vz = load_point_cloud_h5(h5_train, t, grp_key) #each is a (n,1) vector or (1,n) ? CHECK
    Fx = compute_invPwrLaw_features(x,y,z,K,P, period, device) #(d, n)
    Fy = compute_invPwrLaw_features(y,z,x,K,P, period, device)
    Fz = compute_invPwrLaw_features(z,x,y,K,P, period, device)
    A = A + Fx @ Fx.T + Fy @ Fy.T + Fz @ Fz.T #(d,d)
    b = b + Fx @ vx.unsqueeze(1) + Fy @ vy.unsqueeze(1) + Fz @ vz.unsqueeze(1)  #(d,1)

w = torch.linalg.lstsq(A, b).solution #(d,1)
```


```python
R2_all = []
for idx in idxTest:
    x,y,z,vx,vy,vz = load_point_cloud_h5(h5_test, idx, grp_key) 
    Fx = compute_invPwrLaw_features(x,y,z,K,P, period, device) #(d, n)
    Fy = compute_invPwrLaw_features(y,z,x,K,P, period, device)
    Fz = compute_invPwrLaw_features(z,x,y,K,P, period, device)
    #feat_time = time.time()
    F_xyz =  torch.stack([Fx, Fy, Fz], dim=0)  # shape: (3, d, n)
    v_pred = torch.einsum('d,cdn->cn', w.squeeze(), F_xyz).T  # shape: (n, 3)
    v_target = torch.stack([vx, vy, vz], dim=-1)
    R2 = compute_R2(v_pred, v_target)
    R2_all.append(R2)
R2_mean = sum(R2_all)/len(R2_all)

print(f"test set R2={R2_mean:.4f}")
```

    test set R2=0.2695


### Merger tree datasets
- Data download url: https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/trees/
- Store the downloaded data in `TREE_DIR`

For demo purpose, we only download `CS_tree_train.pt`,`CS_tree_val.pt` and `infilling_trees_25k_200_val.pt`


```python
from models.tree_param.model_tree import TreeRegressor, MLPAgg, DeepSet, train_eval_model
from utils.tree_util import dataset_to_dataloader
from torch_geometric.utils import to_networkx

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
```

    Pytorch Geometric is available = True. Return list of PyG.Data = True



```python
tree_dir = "CosmoBench_Trees"
os.makedirs(tree_dir, exist_ok=True)

url = "https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/trees/"
file_names = ['CS_tree_train.pt', 'CS_tree_val.pt', \
              'infilling_trees_25k_200_train.pt', 'infilling_trees_25k_200_val.pt']

for fname in file_names:
    full_url = url + fname
    print(f"Downloading {fname}...")
    subprocess.run(['wget', '-q', '-P', tree_dir, full_url])
```

    Downloading CS_tree_train.pt...
    Downloading CS_tree_val.pt...
    Downloading infilling_trees_25k_200_train.pt...
    Downloading infilling_trees_25k_200_val.pt...



```python
def plot_tree(data):
    G = to_networkx(data, to_undirected=False)
    pos = graphviz_layout(G, prog="dot")
    fig, ax = plt.subplots(dpi=200)
    
    mask_main = torch.isin(data.node_halo_id.flatten(), torch.LongTensor(data.mask_main)) #NOTE: a halo id may appear > 1 if the halo splits
    node_indices = torch.nonzero(mask_main).flatten()

    node_colors = ["tab:pink" if n in node_indices else "skyblue" for n in G.nodes()]
    
    nx.draw(G, pos=pos, with_labels=False, arrows=True, arrowsize=1,
            ax=ax, node_color=node_colors, node_size=3)
```


```python
tree_dir = "CosmoBench_Trees"

trainset_path = f"{tree_dir}/CS_tree_train.pt"
valset_path = f"{tree_dir}/CS_tree_val.pt"
trainset = torch.load(trainset_path) #a list of PyG Data storing node features, edge index, etc
valset = torch.load(valset_path) #a list of PyG Data storing node features, edge index, etc

```


```python
#the pink dots denoting the main branch of the merger tree
sample_tree = trainset[0]
plot_tree(sample_tree)
```

    /tmp/ipykernel_452227/2636837194.py:3: DeprecationWarning: nx.nx_pydot.graphviz_layout depends on the pydot package, which hasknown issues and is not actively maintained. Consider usingnx.nx_agraph.graphviz_layout instead.
    
    See https://github.com/networkx/networkx/issues/5723
      pos = graphviz_layout(G, prog="dot")



    
