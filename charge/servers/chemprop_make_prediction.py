from charge.servers.external_tools.chemprop.chemprop import chemprop
#import chemprop

def get_chemprop_preds(smiles,model_path):

    """
    Generate property predictions for molecules using a pretrained Chemprop model.

    This function loads a pretrained Chemprop model from the specified checkpoint 
    directory and performs inference on a set of SMILES strings. It returns the 
    model’s numerical predictions for the given property.

    Parameters
    ----------
    smiles : list or str
        A SMILES string or a list of SMILES strings representing the molecules to evaluate.
    model_path : str, 
        Path to the trained chemprop model. Must be among the valid_properties, commented out below.

    Returns
    -------
    list
        A nested list of predicted values from the Chemprop model, where each
        inner list corresponds to the predicted property values for a given molecule.

    Notes
    -----
    - The `test_path` and `preds_path` arguments are set to `/dev/null`
      since predictions are made directly from SMILES strings in memory.
    - `num_workers` is set to 0 for compatibility and reproducibility across systems.

    """
   
    #valid_properties = {'10k_density', '10k_hof', 'qm9_alpha','qm9_cv','qm9_gap','qm9_homo','qm9_lumo','qm9_mu','qm9_r2','qm9_zpve','lipo'}
    #if property not in valid_properties:
    #    raise ValueError(
    #        f"Invalid property '{property}'. Must be one of {valid_properties}."
    #    )

    arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    '--num_workers', 0,
    '--checkpoint_dir', model_path]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    model_objects = chemprop.train.load_model(args=args)
    preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)
    return(preds)

#example usage:
#smiles = [['CC(NCC1CO1)C#N'], ['CN(C)c1ccc(cc1)C(=C(C#N)C#N)C(c1ccc(cc1)N(=O)=O)=C1C(=O)c2ccccc2C1=O'], ['CC#CCC(OC(=O)c1ccc(cc1)N(=O)=O)C1(O)C(=O)OCC1(C)C=C']]
#print(get_chemprop_preds(smiles,model_path='Saved_Models/10k_density_OOD/fold_0/'))
