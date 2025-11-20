from __future__ import annotations
import logging
from loguru import logger

try:
    import click
    import torch
    import pandas as pd
    from lightning import pytorch as pl
    from chemprop import data, models, featurizers
    from chemprop.models import MPNN
    HAS_CHEMPROP = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_CHEMPROP = False
    logger.warning(
        "Please install the chemprop support packages to use this module."
        "Install it with: pip install charge[chemprop]",
    )
from typing import List
import numpy as np
import os, sys

def predict_with_chemprop(
    checkpoint_path: str,
    smiles: list[str],
    device: str = "cpu",
    batch_size: int = 50,
    # accelerator: str = "cpu"
) -> list[list[float]]:
    """
    Load a Chemprop v2 model from a checkpoint and predict on a list of SMILES.

    Parameters
    ----------
    checkpoint_path : str
        Path to the trained model checkpoint (.ckpt).
    smiles_list : list[str]
        A list of SMILES strings to run prediction on.
    device : str, optional (default "cpu")
        Device identifier, e.g. "cpu" or "cuda:0".
    batch_size : int, optional
        Batch size for inference.

    Returns
    -------
    List[List[float]]
        Predictions: for each input SMILES, a list of output values (one per task).
    """
    if not HAS_CHEMPROP:
        raise ImportError("Please install the chemprop support packages to use this module.")

    # Load model
    mpnn=MPNN.load_from_file(checkpoint_path)
    #model = model.to(device)
    mpnn.eval()

    # 2) Build datapoints -> dataset -> dataloader
    datapoints = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(datapoints, featurizer=featurizer)
    loader = data.build_dataloader(dset, shuffle=False)  # uses sensible defaults

    # 3) Predict with Lightning Trainer
    with torch.inference_mode():
        trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator=device, devices=1)
        preds_batches = trainer.predict(mpnn, loader)

    preds = np.concatenate(preds_batches, axis=0).tolist()
    return preds


@click.command()
@click.option("--model-dir", envvar="CHEMPROP_BASE_PATH", help="Path to chemprop model")
@click.option("--device", envvar="CHEMPROP_SERVER_DEVICE", default="cpu", help="Device to use for the chemprop server: cpu, cuda")
def main(model_dir: str, device: str):
    if not HAS_CHEMPROP:
        raise ImportError("Please install the chemprop support packages to use this module.")

    chemprop_base_path=model_dir
    if(chemprop_base_path):
        ckpt = os.path.join(chemprop_base_path, "gap/model_0/best.pt")
        smis = ["O=CC12C3CC1CCN23", "CCC", "c1ccccc1O"]
        print(predict_with_chemprop(ckpt, smis, device))
    else:
        print('CHEMPROP_BASE_PATH environment variable not set!')
        sys.exit(2)

if __name__ == "__main__":
    main()
