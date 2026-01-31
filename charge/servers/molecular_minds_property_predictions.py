#!/usr/bin/env python3
"""
Molecular Minds: Multi-Property Predictions for Energetic Materials

Usage:
    from molecular_minds_property_predictions import predict_hof, predict_density
    
    hof = predict_hof('CCO')
    density = predict_density('CCO')
"""

import sys
import os
from typing import Optional, Dict

# Add current directory to path to find molecular_minds module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from molecular_minds package
from molecular_minds import (
    predict_smiles,
    get_default_predictor,
    set_default_model
)

def predict_hof(smiles: str, predictor: Optional[Dict] = None) -> float:
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
    if predictor is None:
        predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['hof_s']

def predict_density(smiles: str, predictor: Optional[Dict] = None) -> float:
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
    if predictor is None:
        predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['density']

def predict_bp(smiles: str, predictor: Optional[Dict] = None) -> float:
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
    if predictor is None:
        predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['bp']

def predict_dh50(smiles: str, predictor: Optional[Dict] = None) -> float:
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
    if predictor is None:
        predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['log(dh50)']

def predict_mp(smiles: str, predictor: Optional[Dict] = None) -> float:
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
    if predictor is None:
        predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['mp']

def predict_vp(smiles: str, predictor: Optional[Dict] = None) -> float:
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
    if predictor is None:
        predictor = get_default_predictor()
    results = predict_smiles(smiles, predictor)
    return results['logvp']

# ===== DIAGNOSTIC SCRIPT =====
if __name__ == "__main__":
    # Optional: explicitly set model path
    # set_default_model('molecular_minds/molecular_minds_model.pth')
    
    # Get SMILES from command line or use default
    if len(sys.argv) > 1:
        smiles = sys.argv[1]
    else:
        smiles = "C1=CC=CC=C1"  # Default: benzene
        print(f"No SMILES provided, using default: {smiles}\n")
    
    print(f"Predicting properties for: {smiles}")
    print("=" * 60)
    
    try:
        hof = predict_hof(smiles)
        density = predict_density(smiles)
        bp = predict_bp(smiles)
        dh50 = predict_dh50(smiles)
        mp = predict_mp(smiles)
        vp = predict_vp(smiles)
        
        print(f"Heat of Formation     : {hof:10.4f} kcal/mol")
        print(f"Density               : {density:10.4f} g/cc")
        print(f"Boiling Point         : {bp:10.2f} °C")
        print(f"log(DH50)             : {dh50:10.4f} cm")
        print(f"Melting Point         : {mp:10.2f} °C")
        print(f"Log Vapor Pressure    : {vp:10.4f} Pa")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo fix this, either:")
        print("  1. Place 'molecular_minds_model.pth' in molecular_minds/ directory")
        print("  2. Set environment variable: export MOLECULAR_MIND_MODEL=/path/to/model.pth")
        print("  3. Call set_default_model('/path/to/model.pth') at the start of your script")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)