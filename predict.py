from glob import glob
import os
import torch
import utils
import anndata as ad
from torch.utils.data import TensorDataset,DataLoader
from pathlib import Path
import numpy as np
from models import MLP
from const import PATH, OUT_PATH


def _predict(model,dl):
    model = model.cuda()
    model.eval()
    yps = []
    for x in dl:
        with torch.no_grad():
            yp = model(x[0].cuda())
            yps.append(yp.detach().cpu().numpy())
    yp = np.vstack(yps)
    return yp
            
def predict(test_data_path,folds,cp='.'):
    y_dim,task = utils.get_y_dim(test_data_path)
    yaml_path=f"{cp}/yaml/mlp_{task}.yaml"
    config = utils.load_yaml(yaml_path)
    te1 = ad.read_h5ad(test_data_path)
    X = te1.X.toarray()
    X = torch.from_numpy(X).float()
    
    te_ds = TensorDataset(X)
    
    yp = 0
    for fold in folds:
        load_path = f'{OUT_PATH}/weights/{task}_fold_{fold}/version_0/checkpoints/*'
        print(load_path)
        ckpt = glob(load_path)[0]
        model_inf = MLP.load_from_checkpoint(ckpt,in_dim=X.shape[1],
                                             out_dim=y_dim,
                                             config=config)
        te_loader = DataLoader(te_ds, batch_size=config.batch_size,num_workers=8,
                        shuffle=False, drop_last=False)
        yp = yp + _predict(model_inf, te_loader)
    return yp/len(folds)

def sanity_check():
    test_data_path = glob('output/pseudo_test/fold_1/*cite*mod2/*mod1.h5ad')
    if len(test_data_path) == 0:
        utils.generate_pseudo_test_data_all()
        test_data_path = glob('output/pseudo_test/fold_1/*cite*mod2/*mod1.h5ad')
    test_data_path = test_data_path[0]
    print(test_data_path)
    
    yp = predict(test_data_path=test_data_path,folds=[2])
    te2 = ad.read_h5ad(test_data_path.replace('mod1','mod2'))
    yt = te2.X.toarray()
    score = ((yp-yt)**2).mean()**0.5
    print(f"VALID RMSE {score:.3f}")
    return yp

if __name__ == '__main__':
    sanity_check()
