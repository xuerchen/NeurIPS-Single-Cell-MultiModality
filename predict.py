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

def copy_weight(task,commit,fold,cp='.'):
    Path(f'{cp}/weights').mkdir(parents=True, exist_ok=True)
    wp = f'{cp}/weights/{task}_{commit}'
    for i in range(3):
        w = f'{wp}_fold_{i}.ckpt'
        if not os.path.exists(w):
            wo = f'{OUT_PATH}/lightning_log/{task}/{commit}/fold_{i}/*/*/*/*.ckpt'
            wo = glob(wo)
            print(wo)
            assert len(wo) == 1
            wo = wo[0]
            cmd = f'cp {wo} {w}'
            os.system(cmd)      
    return f'{wp}_fold_{fold}.ckpt'

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
            
def predict(y_dim,task,yaml_path,test_data_path,folds,cp='.'):
    
    commit = task2commit[task]
    
    config = utils.load_yaml(yaml_path)
    te1 = ad.read_h5ad(test_data_path)
    X = te1.X.toarray()
    X = torch.from_numpy(X).float()
    
    te_ds = TensorDataset(X)
    
    yp = 0
    for fold in folds:
        ckpt = copy_weight(task,commit,fold=fold,cp=cp)
        model_inf = MLP.load_from_checkpoint(ckpt,in_dim=X.shape[1],
                                             out_dim=y_dim,
                                             config=config)
        te_loader = DataLoader(te_ds, batch_size=config.batch_size,num_workers=8,
                        shuffle=False, drop_last=False)
        yp = yp + _predict(model_inf, te_loader)
    return yp/len(folds)

def sanity_check(task2commit,yaml_path):
    test_data_path = glob('output/pseudo_test/fold_1/*multiome*mod2/*mod1.h5ad')[0]
    print(test_data_path)
    yp = predict(task2commit,yaml_path=yaml_path,
            test_data_path=test_data_path,folds=[2])
    te2 = ad.read_h5ad(test_data_path.replace('mod1','mod2'))
    yt = te2.X.toarray()
    score = ((yp-yt)**2).mean()**0.5
    print(f"VALID RMSE {score:.3f}")
    return yp

if __name__ == '__main__':
    task2commit = {
        'GEX2ADT':'1050db0',
        'ADT2GEX':'3cd1f2a',
        'GEX2ATAC':'b3478cd',
        'ATAC2GEX':'e2822b1',
    }
    yaml_path='yaml/mlp_GEX2ADT.yaml'
    sanity_check(task2commit,yaml_path)
