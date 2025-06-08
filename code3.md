#### Task 3: Predicting cosmological parameters from merger trees
- Baseline: message-passing graph neural network (MPNN)


```python
### full dataset with n_samples = 25

batch_size = 128
feat_idx = [0,1,2,3] #(Mass, concentration, vmax, scale)
train_loader, val_loader, test_loader = dataset_to_dataloader(trainset, valset, None,
                                                                batch_size=batch_size,
                                                        normalize=True, feat_idx=feat_idx)

```

    normalizing for mean 0 , std 1 across all trees!
    train_size=14997, val_size=5099, test_size=0
    sampled train data view = Data(x=[363, 4], edge_index=[2, 362], edge_attr=[362, 1], y=[1, 2], num_nodes=363, lh_id=0, mask_main=[94], node_halo_id=[363, 1])


    /mnt/home/thuang/playground/.venv/lib/python3.10/site-packages/torch_geometric/deprecation.py:26: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead
      warnings.warn(out)



```python
#initialize MPNN model
device = "cuda" if torch.cuda.is_available() else "cpu"
node_dim = len(feat_idx)
hid_dim = 16
out_dim = 2
n_layer = 5
model = TreeRegressor(node_dim, hid_dim, out_dim, n_layer, loop_flag=True)
model = model.float() #ensure using float32
model = model.to(device)

```


```python
num_epochs = 10
lr = 5e-3
train_loss_steps, val_loss_eps = train_eval_model(model, train_loader, val_loader, 
                                                mlp_only=False, n_epochs=num_epochs,
                                                lr=lr, target_id=None)
```

    epoch=0, train_loss=0.0219, val_loss=0.0169, R2_om=0.3614, R2_s8=0.2987
    epoch=1, train_loss=0.0077, val_loss=0.0137, R2_om=0.7316, R2_s8=0.1965
    epoch=2, train_loss=0.0065, val_loss=0.0104, R2_om=0.8418, R2_s8=0.3503
    epoch=3, train_loss=0.0042, val_loss=0.0063, R2_om=0.9486, R2_s8=0.5681
    epoch=4, train_loss=0.0028, val_loss=0.0052, R2_om=0.9628, R2_s8=0.6357
    epoch=5, train_loss=0.0022, val_loss=0.0038, R2_om=0.9870, R2_s8=0.7244
    epoch=6, train_loss=0.0021, val_loss=0.0058, R2_om=0.9382, R2_s8=0.6144
    epoch=7, train_loss=0.0020, val_loss=0.0046, R2_om=0.9506, R2_s8=0.6956
    epoch=8, train_loss=0.0018, val_loss=0.0041, R2_om=0.9293, R2_s8=0.7547
    epoch=9, train_loss=0.0018, val_loss=0.0031, R2_om=0.9886, R2_s8=0.7733


#### Task 4: Reconstructing fine-grained merger trees via node classification
- Baseline: MPNN (classifying the augmented virtual nodes in the coarsened tree)


```python
from models.tree_recon.model_infilling import TreeNodeClassifier, train_eval_classifier, eval_classifier
from utils.tree_util import subset_data_features
```


```python
train_path = f"{tree_dir}/infilling_trees_25k_200_train.pt"
val_path = f"{tree_dir}/infilling_trees_25k_200_val.pt"

train_trees =  subset_data_features(torch.load(train_path), feat_idx)
val_trees = subset_data_features(torch.load(val_path), feat_idx)
   
```


```python
out_dim = 2 #binary classification
node_dim = len(feat_idx) #input feature dim
model = TreeNodeClassifier(node_dim, hid_dim, out_dim, n_layer=4, loop_flag=True)
model = model.float()
model = model.to(device)

```


```python
train_loss, val_loss_out, best_val_acc = train_eval_classifier(model, train_trees, val_trees, save_dir=tree_dir, 
                                                               num_epochs=num_epochs, lr=1e-3)

```

    Epoch 0, Train Loss: 0.6470; Val Loss Hold-Out: 0.6456, Val Acc. Hold-Out: 0.6135



```python

```
