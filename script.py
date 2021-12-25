# Dependencies:
# pip: scikit-learn, anndata, scanpy
#
# Python starter kit for the NeurIPS 2021 Single-Cell Competition.
# Parts with `TODO` are supposed to be changed by you.
#
# More documentation:
#
# https://viash.io/docs/creating_components/python/

import logging
import anndata as ad
import sys
from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad',
    'input_test_mod1': 'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod1.h5ad',
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 50,
}
meta = { 'resources_dir': '.' }
## VIASH END
sys.path.append(meta['resources_dir'])
from predict import predict

# TODO: change this to the name of your method
method_id = "simple_mlp"

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

def get_y_dim(data_path):
    if '_cite_' in data_path:
        if 'mod2' in data_path:
            return 13953,"ADT2GEX"
        elif 'rna' in data_path:
            return 134,"GEX2ADT"
        else:
            assert 0
    elif '_multiome_' in data_path:
        if 'mod2' in data_path:
            return 13431,"ATAC2GEX"
        elif 'rna' in data_path:
            return 10000,"GEX2ATAC"
        else:
            assert 0

# TODO: change this to the name of your method
method_id = "simple_mlp"
_,task = get_y_dim(par['input_train_mod1'])

task2commit = {
    'GEX2ADT':'1050db0',
    'ADT2GEX':'3cd1f2a',
    'GEX2ATAC':'b3478cd',
    'ATAC2GEX':'e2822b1',
}
yaml_path=f"{meta['resources_dir']}/yaml/mlp_GEX2ADT.yaml"
y_pred = predict(task2commit,yaml_path=yaml_path,
        test_data_path=par['input_test_mod1'],folds=[0,1,2],cp=meta['resources_dir'])

y_pred = csc_matrix(y_pred)

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': method_id,
    },
)

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")
