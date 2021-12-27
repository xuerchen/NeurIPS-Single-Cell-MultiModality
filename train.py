import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset,DataLoader
from models import MLP
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from pathlib import Path
import utils
import anndata as ad
import numpy as np
import json
from const import PATH, OUT_PATH

def _train(X, y, Xt, yt, enable_ckpt, logger, yaml_path):
    config = utils.load_yaml(yaml_path)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    ymean = torch.mean(y,dim=0,keepdim=True)
    
    tr_ds = TensorDataset(X,y)
    tr_loader = DataLoader(tr_ds, batch_size=config.batch_size,num_workers=8,
                        shuffle=True, drop_last=True)
    
    Xt = torch.from_numpy(Xt).float()
    yt = torch.from_numpy(yt).float()
    te_ds = TensorDataset(Xt,yt)
    te_loader = DataLoader(te_ds, batch_size=config.batch_size,num_workers=8,
                        shuffle=False, drop_last=False)
    
    checkpoint_callback = ModelCheckpoint(monitor='valid_RMSE')
    if enable_ckpt:
        epochs = config.epochs
        cb = [checkpoint_callback]
    else:
        epochs = 1
        cb = None
    
    trainer = pl.Trainer(enable_checkpointing=enable_ckpt, logger=logger, 
                         gpus=1, max_epochs=epochs, 
                         callbacks=cb,
                         progress_bar_refresh_rate=5)
    
    net = MLP(X.shape[1],y.shape[1],ymean,config)
    trainer.fit(net, tr_loader, te_loader)
    
    cp = 'best' if enable_ckpt else None
    yp = trainer.predict(net,te_loader,ckpt_path=cp)
    yp = torch.cat(yp,dim=0)
    
    score = ((yp-yt)**2).mean()**0.5
    print(f"VALID RMSE {score:.3f}")
    del trainer
    return score,yp.detach().numpy()


def train(task,dummy_run=True):
    yaml_path = f'yaml/mlp_{task}.yaml'
    yps = []
    scores = []
    
    tr1,tr2,te1,te2 = utils.get_data_paths(task,phase='phase2',path=f'{PATH}/output')
    tr1 = ad.read_h5ad(tr1)
    tr2 = ad.read_h5ad(tr2)

    msgs = {}
    for fold in range(3):

        if not dummy_run:
            run_name = f"{task}_fold_{fold}"
            save_path = f'{OUT_PATH}/weights/{run_name}'
            Path(save_path).mkdir(parents=True, exist_ok=True)   

        X,y,Xt,yt = utils.split(tr1, tr2, fold)
        run_name = f'fold_{fold}'
        if dummy_run:
            logger = None
        else:
            logger = TensorBoardLogger(save_path, name='') 
        
        if dummy_run:
            X = X[:512]
            y = y[:512]
            Xt = Xt[:512]
            yt = yt[:512]
            enable_ckpt = False
        else:
            enable_ckpt = True
        
        score, yp = _train(X, y, Xt, yt, enable_ckpt, logger, yaml_path)
        #yp = np.mean(y,axis=0,keepdims=True)*np.ones_like(yt)
        #score = ((yp-yt)**2).mean()**0.5
        yps.append(yp)
        scores.append(score)
        msg = f"{task} Fold {fold} RMSE {score:.3f}"
        msgs[f'Fold {fold}'] = f'{score:.3f}'
        print(msg)


    yp = np.concatenate(yps)
    score = np.mean(scores)
    msgs['Overall'] = f'{score:.3f}'
    print('Overall', f'{score:.3f}')

    
if __name__ == '__main__':
    dummy_run = False
    for task in ['GEX2ADT','ADT2GEX','ATAC2GEX']:
        train(task=task,dummy_run=dummy_run)
