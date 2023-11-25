from jarvis.db.figshare import data
import pandas as pd
import featurize
from jarvis.core.atoms import Atoms
from matminer.featurizers.site import AverageBondLength, AverageBondAngle, LocalPropertyDifference, SiteElementalProperty, CoordinationNumber
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (ElementFraction, TMetalFraction, Stoichiometry, BandCenter, OxidationStates,
                                              ElectronegativityDiff, IonProperty, ElectronAffinity, AtomicOrbitals, ValenceOrbital)
from matminer.featurizers.structure import (SiteStatsFingerprint, MinimumRelativeDistances, StructureComposition)
from matminer.featurizers.conversions import StructureToOxidStructure
from pymatgen.analysis.local_env import CrystalNN

import pymatgen as mp
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import os
import argparse
import logging
from tqdm import tqdm 
from pymatgen.analysis.local_env import VoronoiNN
parser = argparse.ArgumentParser(description='run matminer featurizer on QMOF dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--cif_dir', help='input data directory', default="./MOFs_oms", type=str)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--output_dir', help='path to the save output features', default="./features", type=str)
parser.add_argument('--start', help='start index', default=0, type=int,required=False)
parser.add_argument('--end', help='end index', default=0, type=int,required=False)
args =  parser.parse_args()

def featurize(args):
    cif_path = args.cif_dir
    cif_files = os.listdir(cif_path)
    args =  parser.parse_args()

    featurizer = MultipleFeaturizer([
        StructureComposition(ElementFraction()), # Class to calculate the atomic fraction of each element in a composition.
        StructureComposition(TMetalFraction()), # Class to calculate fraction of magnetic transition metals in a composition
        StructureComposition(Stoichiometry()), # Calculate norms of stoichiometric attributes.
        StructureComposition(BandCenter()), # Estimation of absolute position of band center using electronegativity
        StructureComposition(OxidationStates()), # Statistics about the oxidation states for each specie.
        StructureComposition(IonProperty(fast=True)), # Ionic property attributes. Similar to ElementProperty
        StructureComposition(ElectronAffinity()), # Calculate average electron affinity times formal charge of anion elements.
        StructureComposition(ElectronegativityDiff()), # Features from electronegativity differences between anions and cations.
        StructureComposition(AtomicOrbitals()), # Determine HOMO/LUMO features based on a composition.
        StructureComposition(ValenceOrbital(props=['frac'])), # Attributes of valence orbital shells
        SiteStatsFingerprint(AverageBondAngle(VoronoiNN()), stats = ("minimum", "maximum", "mean", "std_dev")), # Determines the average bond length between one specific site
        SiteStatsFingerprint(AverageBondLength(VoronoiNN()), stats = ("minimum", "maximum", "mean", "std_dev")), # Determines the average bond angles of a specific site with
        SiteStatsFingerprint(LocalPropertyDifference(), stats = ("minimum", "maximum", "mean", "std_dev")), # Differences in elemental properties between site and its neighboring sites.
        SiteStatsFingerprint(SiteElementalProperty(), stats = ("minimum", "maximum", "mean", "std_dev")), # Elemental properties of atom on a certain site
        SiteStatsFingerprint(CoordinationNumber(), stats = ("minimum", "maximum", "mean", "std_dev")), # Number of first nearest neighbors of a site.
        # MinimumRelativeDistances(), # Determines the relative distance of each site to its closest neighbor.

    ])
    features = []
    featurizer.set_n_jobs(12)
    start = 0  if not args.start else args.start
    end = len(cif_files) if not args.end else args.end
    for cif in tqdm(cif_files[start:end], desc="Featurize mof"):
        cif_struc = mp.core.Structure.from_file(os.path.join(cif_path, cif))
        sto = StructureToOxidStructure(max_sites = -1)
        cif_struc_oxi = sto.featurize(cif_struc)[0]
        cif_feature = featurizer.featurize(cif_struc_oxi)
        features.append(cif_feature)
    feature_labels = featurizer.feature_labels()
    df = pd.DataFrame(features, index=cif_files[start:end], columns = feature_labels)
    return df

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    df_fea = featurize(args)
    output_csv = f"matminer_{args.start}_{args.start + len(df_fea)}.csv"
    output_csv = os.path.join(args.output_dir, output_csv)
    df_fea.to_csv(output_csv)
    logging.info(f" File saved to : {output_csv}")
