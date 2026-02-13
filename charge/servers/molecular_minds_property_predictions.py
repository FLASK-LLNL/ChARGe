#!/usr/bin/env python3
"""
Molecular Minds: Multi-Property Predictions for Materials

Usage:
    from molecular_minds_property_predictions import predict_hof, predict_density
    
    hof = predict_hof('CCO')
    density = predict_density('CCO')
"""

import sys
import os
from typing import Optional, Dict
import argparse

from fastmcp import FastMCP
mcp = FastMCP("Molecular Minds Property Predictor", json_response=True)

# Add path to find Molecular_Minds module
current_dir = os.path.dirname(os.path.abspath(__file__))          
parent_dir = os.path.dirname(current_dir)                         
grandparent_dir = os.path.dirname(parent_dir)                     
molecular_minds_path = os.path.join(grandparent_dir, "examples") 

if molecular_minds_path not in sys.path:
    sys.path.insert(0, molecular_minds_path)

# Import from Molecular_Minds package
from Molecular_Minds import (
    predict_smiles,
    get_default_predictor,
    set_default_model
)
import Molecular_Minds.molecular_minds_predictor

@mcp.tool()
def predict_hof(smiles: str) -> float:
    """
    Predict Heat of Formation (hof_s) for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.
        
    Returns:
        Predicted hof_s value in kcal/mol
        
    Example:
        >>> hof = predict_hof('CCO')
        >>> print(f"HOF: {hof:.2f} kcal/mol")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['hof_s']

@mcp.tool()
def predict_density(smiles: str) -> float:
    """
    Predict density for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.
        
    Returns:
        Predicted density in g/cc
        
    Example:
        >>> density = predict_density('CCO')
        >>> print(f"Density: {density:.3f} g/cc")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['density']

@mcp.tool()
def predict_bp(smiles: str) -> float:
    """
    Predict boiling point (bp) for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.
        
    Returns:
        Predicted boiling point in °C
        
    Example:
        >>> bp = predict_bp('CCO')
        >>> print(f"Boiling Point: {bp:.1f} °C")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['bp']

@mcp.tool()
def predict_dh50(smiles: str) -> float:
    """
    Predict log(dh50) for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.
        
    Returns:
        Predicted log(dh50) value in cm
        
    Example:
        >>> dh50 = predict_dh50('CCO')
        >>> print(f"log(DH50): {dh50:.3f}")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['log(dh50)']

@mcp.tool()
def predict_mp(smiles: str) -> float:
    """
    Predict melting point (mp) for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.
        
    Returns:
        Predicted melting point in °C
        
    Example:
        >>> mp = predict_mp('CCO')
        >>> print(f"Melting Point: {mp:.1f} °C")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['mp']

@mcp.tool()
def predict_vp(smiles: str) -> float:
    """
    Predict log vapor pressure (logvp) for a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        predictor: Optional. Dictionary returned by load_model(). If None, uses default model.
        
    Returns:
        Predicted logvp value in Pa
        
    Example:
        >>> vp = predict_vp('CCO')
        >>> print(f"Log Vapor Pressure: {vp:.3f} log(Pa)")
    """
    predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['logvp']

# ===== SCRIPT =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        type=str,
        help="Path to model checkpoint path"
    )
    args = parser.parse_args()
    model_path = args.checkpoint_path
    Molecular_Minds.molecular_minds_predictor.DEFAULT_MODEL_PATH = model_path
mcp.run(transport="sse")