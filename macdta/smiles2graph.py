"""
SMILES to Graph conversion utilities for PMHGT-DTA
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # Atomic numbers 1-118
    'chirality': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'num_hs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'num_radical_electrons': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}


def one_hot_encoding(x, allowable_set):
    """One-hot encoding of categorical variables"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    """
    Get atom features as a list
    """
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetChiralTag(), ATOM_FEATURES['chirality'])
    features += one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot_encoding(atom.GetNumRadicalElectrons(), ATOM_FEATURES['num_radical_electrons'])
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
    features += one_hot_encoding(atom.IsInRing(), ATOM_FEATURES['is_in_ring'])
    return features


def get_bond_features(bond):
    """
    Get bond features as a list
    """
    bond_type = bond.GetBondType()
    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    return features


def smile2graph4drugood(smiles):
    """
    Convert SMILES string to graph representation
    
    Args:
        smiles: SMILES string
        
    Returns:
        c_size: number of atoms
        edge_index: edge indices as numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    c_size = mol.GetNumAtoms()
    
    # Get edges
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # Add reverse edge for undirected graph
    
    if len(edges) == 0:
        # Single atom molecule
        edge_index = np.array([[0], [0]])
    else:
        edge_index = np.array(edges).T
    
    return c_size, edge_index


def smiles_to_graph_data(smiles, include_features=True):
    """
    Convert SMILES to PyTorch Geometric Data object
    
    Args:
        smiles: SMILES string
        include_features: whether to include atom/bond features
        
    Returns:
        Data object with node features, edge index, and edge features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    if include_features:
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        x = torch.tensor(atom_features, dtype=torch.float)
    else:
        x = torch.ones((mol.GetNumAtoms(), 1), dtype=torch.float)
    
    # Get edges and edge features
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_features = get_bond_features(bond)
        
        # Add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)
    
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def batch_smiles_to_graphs(smiles_list):
    """
    Convert a batch of SMILES to graph data objects
    
    Args:
        smiles_list: list of SMILES strings
        
    Returns:
        dict mapping SMILES to Data objects
    """
    graph_dict = {}
    for smiles in smiles_list:
        data = smiles_to_graph_data(smiles)
        if data is not None:
            graph_dict[smiles] = data
    return graph_dict


if __name__ == "__main__":
    # Test the functions
    test_smiles = "CCO"
    c_size, edge_index = smile2graph4drugood(test_smiles)
    print(f"Molecule size: {c_size}")
    print(f"Edge index shape: {edge_index.shape}")
    
    data = smiles_to_graph_data(test_smiles)
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")

