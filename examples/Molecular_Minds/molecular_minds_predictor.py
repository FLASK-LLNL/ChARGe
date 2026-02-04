#!/usr/bin/env python3
"""
Molecular Minds: Multi-Property Predictions for Materials

Usage:
    from molecular_minds_predictor import load_model, predict_smiles
    
    predictor = load_model('path/to/model.pth')
    results = predict_smiles('CCO', predictor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Optional, Union
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from functools import lru_cache

try:
    from torch_geometric.utils import scatter_mean
except ImportError:
    from torch_scatter import scatter_mean

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Crippen
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from sklearn.preprocessing import StandardScaler

# ===== CACHED MODEL LOADER =====

# Default model path - can be overridden by environment variable
DEFAULT_MODEL_PATH = os.environ.get(
    'MOLECULAR_MIND_MODEL',
    os.path.join(os.path.dirname(__file__), 'molecular_minds_model.pth')
)

_global_predictor = None

def set_default_model(model_path: str):
    """
    Set the default model path for use with simplified prediction functions.
    
    Args:
        model_path: Path to the .pth model file
    """
    global _global_predictor
    _global_predictor = load_model(model_path)
    return _global_predictor

def get_default_predictor():
    """
    Get the default predictor, loading it if necessary.
    Caches the model after first load.
    
    Returns:
        Dictionary containing the model and metadata
    """
    global _global_predictor
    
    if _global_predictor is None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {DEFAULT_MODEL_PATH}\n"
            )
        _global_predictor = load_model(DEFAULT_MODEL_PATH)
    
    return _global_predictor

# ===== MODEL ARCHITECTURE CLASSES =====

class DistanceMessagePassing(MessagePassing):
    """Distance-enhanced message passing"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, pos):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
    
    def message(self, x_i, x_j, edge_attr, pos_i, pos_j):
        distance = torch.norm(pos_i - pos_j, dim=1, keepdim=True)
        distance_features = torch.cat([
            distance,
            1.0 / (1.0 + distance),
            torch.exp(-distance)
        ], dim=1)
        
        message = torch.cat([x_i, x_j, distance_features], dim=1)
        return self.mlp(message)

class LearnedFeatureExtractor(nn.Module):
    """Extract learned features from atom embeddings"""
    
    def __init__(self, atom_embed_dim: int, learned_feature_dim: int = 64):
        super().__init__()
        self.learned_feature_dim = learned_feature_dim
        
        self.attention_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=atom_embed_dim,
            num_heads=self.attention_heads,
            dropout=0.1,
            batch_first=False
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(atom_embed_dim, atom_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(atom_embed_dim // 2, learned_feature_dim)
        )
        
        self.pool_projections = nn.ModuleList([
            nn.Linear(atom_embed_dim, learned_feature_dim // 4) for _ in range(4)
        ])
        
    def forward(self, atom_embeddings: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        batch_size = batch_indices.max().item() + 1
        learned_features_list = []
        
        for batch_idx in range(batch_size):
            atom_mask = (batch_indices == batch_idx)
            mol_atoms = atom_embeddings[atom_mask]
            
            if mol_atoms.shape[0] == 0:
                learned_features_list.append(torch.zeros(self.learned_feature_dim, device=atom_embeddings.device))
                continue
            
            mol_atoms_t = mol_atoms.unsqueeze(1)
            attended_atoms, _ = self.attention(mol_atoms_t, mol_atoms_t, mol_atoms_t)
            attended_atoms = attended_atoms.squeeze(1)
            
            attention_weights = F.softmax(torch.sum(attended_atoms * mol_atoms, dim=1), dim=0)
            weighted_mean = torch.sum(attended_atoms * attention_weights.unsqueeze(1), dim=0)
            max_pooled, _ = torch.max(attended_atoms, dim=0)
            mean_pooled = torch.mean(attended_atoms, dim=0)
            std_pooled = torch.std(attended_atoms, dim=0)
            
            pooled_results = [weighted_mean, max_pooled, mean_pooled, std_pooled]
            projected_features = []
            
            for pooled_result, projection in zip(pooled_results, self.pool_projections):
                projected = projection(pooled_result)
                projected_features.append(projected)
            
            mol_learned_features = torch.cat(projected_features, dim=0)
            learned_features_list.append(mol_learned_features)
        
        learned_features = torch.stack(learned_features_list, dim=0)
        return learned_features

class MultiPropertyMolecularMind(nn.Module):
    """Multi-Property MolecularMind with property-specific gates and layers"""
    
    def __init__(self, atom_feature_dim: int, molecular_feature_dim: int, 
                 property_names: List[str], hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.1, gate_regularization: float = 0.01,
                 learned_feature_dim: int = 64):
        super().__init__()
        
        self.property_names = property_names
        self.num_properties = len(property_names)
        
        self.property_atom_gates = nn.ParameterDict({
            prop: nn.Parameter(torch.ones(atom_feature_dim)) for prop in property_names
        })
        self.property_molecular_gates = nn.ParameterDict({
            prop: nn.Parameter(torch.ones(molecular_feature_dim)) for prop in property_names
        })
        self.property_learned_gates = nn.ParameterDict({
            prop: nn.Parameter(torch.ones(learned_feature_dim)) for prop in property_names
        })
        
        self.gate_regularization = gate_regularization
        
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        
        self.mp_layers = nn.ModuleList([
            DistanceMessagePassing(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.learned_feature_extractor = LearnedFeatureExtractor(
            atom_embed_dim=hidden_dim,
            learned_feature_dim=learned_feature_dim
        )
        
        self.property_molecular_embedding = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(molecular_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout // 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            ) for prop in property_names
        })
        
        self.property_learned_embedding = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(learned_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout // 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            ) for prop in property_names
        })
        
        combined_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        self.property_fusion = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ) for prop in property_names
        })
        
        self.property_heads = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            ) for prop in property_names
        })
    
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        x = self.atom_embedding(batch.x)
        
        for mp_layer, norm in zip(self.mp_layers, self.layer_norms):
            residual = x
            x = mp_layer(x, batch.edge_index, batch.edge_attr, batch.pos)
            x = norm(x + residual)
        
        explicit_atom_representation = scatter_mean(x, batch.batch, dim=0)
        learned_features = self.learned_feature_extractor(x, batch.batch)
        
        predictions = {}
        
        for prop_name in self.property_names:
            atom_gates = torch.sigmoid(self.property_atom_gates[prop_name])
            molecular_gates = torch.sigmoid(self.property_molecular_gates[prop_name])
            learned_gates = torch.sigmoid(self.property_learned_gates[prop_name])
            
            gated_molecular_features = batch.molecular_features * molecular_gates.unsqueeze(0)
            gated_learned_features = learned_features * learned_gates.unsqueeze(0)
            
            molecular_representation = self.property_molecular_embedding[prop_name](gated_molecular_features)
            learned_representation = self.property_learned_embedding[prop_name](gated_learned_features)
            
            combined_representation = torch.cat([
                explicit_atom_representation,
                molecular_representation,
                learned_representation
            ], dim=1)
            
            fused_representation = self.property_fusion[prop_name](combined_representation)
            predictions[prop_name] = self.property_heads[prop_name](fused_representation)
        
        return predictions

class PropertyScaler:
    """Handle property-specific normalization and denormalization"""
    
    def __init__(self):
        self.scalers = {}
        self.fitted_properties = set()
    
    def inverse_transform(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize predictions back to original scale"""
        denormalized = {}
        for prop_name, pred_tensor in predictions.items():
            if prop_name in self.fitted_properties:
                scaler = self.scalers[prop_name]
                pred_np = pred_tensor.detach().cpu().numpy()
                denorm_np = scaler.inverse_transform(pred_np.reshape(-1, 1)).flatten()
                denormalized[prop_name] = torch.tensor(denorm_np, device=pred_tensor.device).reshape(pred_tensor.shape)
            else:
                denormalized[prop_name] = pred_tensor
        
        return denormalized

# ===== FEATURE CALCULATION FUNCTIONS =====

def smiles_to_3d_coords(smiles: str, max_attempts: int = 5) -> Optional[np.ndarray]:
    """Generate 3D coordinates from SMILES using RDKit ETKDG"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        for attempt in range(max_attempts):
            try:
                confId = AllChem.EmbedMolecule(mol, randomSeed=42 + attempt)
                if confId != -1:
                    AllChem.MMFFOptimizeMolecule(mol, confId=confId)
                    conformer = mol.GetConformer(confId)
                    return conformer.GetPositions()
            except:
                continue
        
        return None
    except:
        return None


def calculate_mol_density_davis2025(mol, coords):
    """Calculate MolDensity descriptor from Davis et al. 2025"""
    try:
        mol_weight = Descriptors.MolWt(mol)
        atoms = mol.GetAtoms()
        n_atoms = len(atoms)
        
        if n_atoms == 0 or len(coords) != n_atoms:
            return 1.0
        
        vdw_radii = []
        for atom in atoms:
            atomic_num = atom.GetAtomicNum()
            try:
                vdw_radius = Chem.GetPeriodicTable().GetRvdw(atomic_num)
                vdw_radii.append(vdw_radius)
            except:
                fallback_radii = {
                    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
                    15: 1.80, 16: 2.58, 17: 1.75, 35: 1.85, 53: 1.98,
                }
                vdw_radii.append(fallback_radii.get(atomic_num, 1.5 + (atomic_num - 1) * 0.05))
        
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        max_radius = max(vdw_radii)
        min_coords -= max_radius
        max_coords += max_radius
        
        box_size = max_coords - min_coords
        box_volume = np.prod(box_size)
        
        n_samples = min(1000, max(200, n_atoms * 50))
        
        np.random.seed(42 + n_atoms)
        random_points = np.random.random((n_samples, 3)) * box_size + min_coords
        
        points_inside = 0
        for point in random_points:
            distances_sq = np.sum((coords - point)**2, axis=1)
            for i, vdw_radius in enumerate(vdw_radii):
                if distances_sq[i] <= vdw_radius**2:
                    points_inside += 1
                    break
        
        if n_samples > 0:
            molecular_volume_ang3 = box_volume * (points_inside / n_samples)
        else:
            molecular_volume_ang3 = 50.0
        
        avogadro = 6.02214076e23
        molecular_volume_ml_per_mol = molecular_volume_ang3 * 1e-24 * avogadro
        mol_density = mol_weight / molecular_volume_ml_per_mol if molecular_volume_ml_per_mol > 0 else 1.0
        
        if mol_density < 0.1 or mol_density > 10.0:
            mol_density = 1.0
        
        return mol_density
        
    except:
        return 1.0

def calculate_clean_atom_features(atom, mol) -> List[float]:
    """Calculate clean atom-level features"""
    features = []
    
    ATOM_FEATURES = {
        'atomic_num': list(range(1, 19)) + [35, 53],
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-2, -1, 0, 1, 2],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [0, 1, 2, 3, 4, 5]
    }
    
    atom_data = {
        'atomic_num': atom.GetAtomicNum(),
        'degree': atom.GetTotalDegree(),
        'formal_charge': atom.GetFormalCharge(),
        'chiral_tag': int(atom.GetChiralTag()),
        'num_Hs': atom.GetTotalNumHs(),
        'hybridization': int(atom.GetHybridization())
    }
    
    for key, choices in ATOM_FEATURES.items():
        value = atom_data[key]
        one_hot = [0] * (len(choices) + 1)
        if value in choices:
            one_hot[choices.index(value)] = 1
        else:
            one_hot[-1] = 1
        features.extend(one_hot)
    
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(atom.GetMass() * 0.01)
    
    features.append(1 if atom.IsInRing() else 0)
    ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings() 
                  if atom.GetIdx() in ring]
    min_ring_size = min(ring_sizes) if ring_sizes else 0
    features.append(min_ring_size * 0.1)
    
    return features

def calculate_clean_molecular_features(mol, coords) -> List[float]:
    """Calculate clean molecular-level features"""
    features = []
    
    try:
        mol_no_hs = Chem.RemoveHs(mol)
        
        features.extend([
            calculate_mol_density_davis2025(mol, coords),
            Descriptors.MolWt(mol_no_hs) * 0.001,
            Descriptors.MolLogP(mol_no_hs) * 0.1,
            rdMolDescriptors.CalcTPSA(mol_no_hs) * 0.01,
            Descriptors.MolMR(mol_no_hs) * 0.01,
            rdMolDescriptors.CalcNumHBD(mol_no_hs),
            rdMolDescriptors.CalcNumHBA(mol_no_hs),
            rdMolDescriptors.CalcNumRotatableBonds(mol_no_hs)
        ])
        
        masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
        center_of_mass = np.average(coords, axis=0, weights=masses)
        centered_coords = coords - center_of_mass
        
        rg_sq = np.sum(masses[:, np.newaxis] * (centered_coords**2)) / np.sum(masses)
        radius_of_gyration = np.sqrt(rg_sq) * 0.1
        features.append(radius_of_gyration)
        
        inertia_tensor = np.zeros((3, 3))
        for pos, mass in zip(centered_coords, masses):
            r_sq = np.sum(pos**2)
            inertia_tensor += mass * (r_sq * np.eye(3) - np.outer(pos, pos))
        
        eigenvals = np.linalg.eigvals(inertia_tensor)
        eigenvals = np.sort(eigenvals)[::-1]
        
        if eigenvals[0] > 1e-6:
            moment_ratio = eigenvals[2] / eigenvals[0]
            asphericity = (eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])) / eigenvals[0]
        else:
            moment_ratio = 0.0
            asphericity = 0.0
        
        features.extend([moment_ratio, asphericity])
        
        ring_info = mol_no_hs.GetRingInfo()
        aromatic_rings = sum(1 for ring in ring_info.AtomRings() 
                           if all(mol_no_hs.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        total_rings = len(ring_info.AtomRings())
        
        features.extend([
            total_rings,
            aromatic_rings,
            total_rings - aromatic_rings,
            total_rings / mol_no_hs.GetNumBonds() if mol_no_hs.GetNumBonds() > 0 else 0
        ])
        
        atom_counts = {'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Cl': 0, 'other': 0}
        for atom in mol_no_hs.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_counts:
                atom_counts[symbol] += 1
            else:
                atom_counts['other'] += 1
        
        total_heavy = sum(atom_counts.values())
        if total_heavy > 0:
            for count in atom_counts.values():
                features.append(count / total_heavy)
        else:
            features.extend([0.0] * 6)
        
        try:
            from rdkit.Chem import QED
            qed_score = QED.qed(mol_no_hs)
        except:
            qed_score = 0.5
        
        try:
            fraction_csp3 = rdMolDescriptors.CalcFractionCsp3(mol_no_hs)
        except:
            fraction_csp3 = 0.5
        
        lipinski_violations = 0
        mol_wt = Descriptors.MolWt(mol_no_hs)
        mol_logp = Descriptors.MolLogP(mol_no_hs)
        hbd = rdMolDescriptors.CalcNumHBD(mol_no_hs)
        hba = rdMolDescriptors.CalcNumHBA(mol_no_hs)
        
        if mol_wt > 500: lipinski_violations += 1
        if mol_logp > 5: lipinski_violations += 1
        if hbd > 5: lipinski_violations += 1
        if hba > 10: lipinski_violations += 1
        
        features.extend([
            qed_score,
            fraction_csp3,
            lipinski_violations * 0.25,
            1 if lipinski_violations == 0 else 0,
            min(mol_wt / 500.0, 2.0)
        ])
        
    except:
        features = [1.0] + [0.0] * 29
    
    while len(features) < 30:
        features.append(0.0)
    
    return features[:30]

def get_clean_bond_features(bond) -> List[float]:
    """Generate clean bond features"""
    if bond is None:
        return [0] * 15
    
    features = []
    
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                 Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type = bond.GetBondType()
    for bt in bond_types:
        features.append(1 if bond_type == bt else 0)
    
    features.extend([
        1 if bond.GetIsConjugated() else 0,
        1 if bond.IsInRing() else 0,
        1 if bond.GetIsAromatic() else 0
    ])
    
    bond_order_map = {
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5
    }
    features.append(bond_order_map.get(bond_type, 1.0) * 0.1)
    
    stereo_types = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
                   Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
    stereo = bond.GetStereo()
    for st in stereo_types:
        features.append(1 if stereo == st else 0)
    
    electronegativity_map = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                           15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
    
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    begin_en = electronegativity_map.get(begin_atom.GetAtomicNum(), 2.5)
    end_en = electronegativity_map.get(end_atom.GetAtomicNum(), 2.5)
    
    features.extend([
        abs(begin_en - end_en) * 0.1,
        (begin_en + end_en) * 0.05
    ])
    
    if bond.IsInRing():
        mol = bond.GetOwningMol()
        ring_info = mol.GetRingInfo()
        bond_rings = [ring for ring in ring_info.BondRings() 
                     if bond.GetIdx() in ring]
        if bond_rings:
            min_ring_size = min(len(ring) for ring in bond_rings)
            features.append(min_ring_size * 0.1)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    return features[:15]

def smiles_to_prediction_graph_data(smiles: str) -> Optional[Data]:
    """Convert SMILES to graph data for prediction"""
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        coords = smiles_to_3d_coords(smiles)
        if coords is None:
            return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = calculate_clean_atom_features(atom, mol)
            atom_features.append(atom_feat)
        
        molecular_features = calculate_clean_molecular_features(mol, coords)
        
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            edge_indices.extend([[start, end], [end, start]])
            
            bond_feat = get_clean_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])
        
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        pos = torch.tensor(coords, dtype=torch.float)
        molecular_features = torch.tensor([molecular_features], dtype=torch.float)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            molecular_features=molecular_features,
            smiles=smiles
        )
        
    except:
        return None

# ===== MAIN API FUNCTIONS =====

def load_model(model_path: str):
    """
    Load a trained MolecularMind model.
    
    Args:
        model_path: Path to the .pth model file
        
    Returns:
        Dictionary containing the model and metadata
        
    Raises:
        ValueError: If model file cannot be loaded
    """
    import sys
    import pickle
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fix for pickle deserialization of PropertyScaler
    if '__main__' in sys.modules:
        sys.modules['__main__'].PropertyScaler = PropertyScaler
    
    # Custom unpickler to handle __main__ references
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__':
                if name == 'PropertyScaler':
                    return PropertyScaler
            return super().find_class(module, name)
    
    try:
        # Try standard loading first
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        # If that fails, try custom unpickler
        try:
            with open(model_path, 'rb') as f:
                checkpoint = CustomUnpickler(f).load()
        except Exception as e2:
            raise ValueError(f"Failed to load model file: {e2}")
    
    property_names = checkpoint['property_names']
    property_scaler = checkpoint['property_scaler']
    
    atom_feature_dim = checkpoint['atom_feature_dim']
    molecular_feature_dim = checkpoint['molecular_feature_dim']
    learned_feature_dim = checkpoint['learned_feature_dim']
    model_config = checkpoint['model_config']
    
    model = MultiPropertyMolecularMind(
        atom_feature_dim=atom_feature_dim,
        molecular_feature_dim=molecular_feature_dim,
        property_names=property_names,
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        gate_regularization=model_config['gate_regularization'],
        learned_feature_dim=learned_feature_dim
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'property_scaler': property_scaler,
        'property_names': property_names,
        'device': device,
        'model_path': model_path,
        'training_stats': checkpoint.get('training_stats', {})
    }

def predict_smiles(smiles: str, predictor: Dict) -> Dict[str, float]:
    """
    Predict properties for a single SMILES string.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Dictionary returned by load_model()
        
    Returns:
        Dictionary mapping property names to predicted values
        
    Raises:
        ValueError: If SMILES cannot be processed
    """
    graph_data = smiles_to_prediction_graph_data(smiles)
    if graph_data is None:
        raise ValueError(f"Failed to process SMILES: {smiles}")
    
    model = predictor['model']
    property_scaler = predictor['property_scaler']
    device = predictor['device']
    property_names = predictor['property_names']
    
    batch = Batch.from_data_list([graph_data]).to(device)
    
    with torch.no_grad():
        predictions = model(batch)
        denorm_predictions = property_scaler.inverse_transform(predictions)
    
    results = {}
    for prop_name in property_names:
        value = denorm_predictions[prop_name].cpu().item()
        results[prop_name] = value
    
    return results

def predict_smiles_batch(smiles_list: List[str], predictor: Dict, batch_size: int = 32) -> List[Dict[str, float]]:
    """
    Predict properties for multiple SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        predictor: Dictionary returned by load_model()
        batch_size: Number of molecules to process at once
        
    Returns:
        List of dictionaries, one per input SMILES
    """
    model = predictor['model']
    property_scaler = predictor['property_scaler']
    device = predictor['device']
    property_names = predictor['property_names']
    
    results = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        graph_data_list = []
        valid_indices = []
        
        for j, smiles in enumerate(batch_smiles):
            graph_data = smiles_to_prediction_graph_data(smiles)
            if graph_data is not None:
                graph_data_list.append(graph_data)
                valid_indices.append(j)
        
        if not graph_data_list:
            for smiles in batch_smiles:
                results.append({prop: float('nan') for prop in property_names})
            continue
        
        batch = Batch.from_data_list(graph_data_list).to(device)
        
        with torch.no_grad():
            predictions = model(batch)
            denorm_predictions = property_scaler.inverse_transform(predictions)
        
        batch_results = [{prop: float('nan') for prop in property_names} for _ in batch_smiles]
        
        for k, valid_idx in enumerate(valid_indices):
            for prop_name in property_names:
                value = denorm_predictions[prop_name][k].cpu().item()
                batch_results[valid_idx][prop_name] = value
        
        results.extend(batch_results)
    
    return results

def predict_csv(input_csv: str, predictor: Dict, smiles_col: str = 'smiles', 
                output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Predict properties for molecules in a CSV file.
    
    Args:
        input_csv: Path to input CSV file
        predictor: Dictionary returned by load_model()
        smiles_col: Name of column containing SMILES strings
        output_csv: Optional path for output CSV (auto-generated if None)
        
    Returns:
        DataFrame with predictions added as new columns
    """
    df = pd.read_csv(input_csv)
    
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in CSV")
    
    smiles_list = df[smiles_col].fillna('').astype(str).tolist()
    
    predictions = predict_smiles_batch(smiles_list, predictor)
    
    property_names = predictor['property_names']
    for prop_name in property_names:
        df[f'predicted_{prop_name}'] = [pred[prop_name] for pred in predictions]
    
    if output_csv is None:
        input_dir = os.path.dirname(os.path.abspath(input_csv))
        input_basename = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(input_dir, f"{input_basename}_predictions.csv")
    
    df.to_csv(output_csv, index=False)
    
    return df

def get_property(results: Dict[str, float], property_name: str) -> float:
    """
    Extract a specific property from prediction results.
    
    Args:
        results: Dictionary returned by predict_smiles()
        property_name: Name of the property to extract
        
    Returns:
        Predicted value for the property
        
    Raises:
        KeyError: If property not in results
    """
    return results[property_name]

def get_available_properties(predictor: Dict) -> List[str]:
    """
    Get list of properties the model can predict.
    
    Args:
        predictor: Dictionary returned by load_model()
        
    Returns:
        List of property names
    """
    return predictor['property_names']

def get_model_info(predictor: Dict) -> Dict:
    """
    Get information about the loaded model.
    
    Args:
        predictor: Dictionary returned by load_model()
        
    Returns:
        Dictionary with model metadata
    """
    return {
        'properties': predictor['property_names'],
        'device': str(predictor['device']),
        'model_path': predictor['model_path'],
        'training_stats': predictor['training_stats']
    }