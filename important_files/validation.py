from utils import utils
import os, re
from rdkit import Chem

if __name__ == "__main__":
    data = open("chebi_smiles_sem_classes.txt", 'r').read().split('\n')[:-1]
    data = [Chem.MolFromSmiles(this) for this in data]

    files = os.listdir("smiles")
    files_range = range(0, 10*len(files), 10)
    
    file_to_write = open("scores.txt", "w")

    for i in files_range:
        with open("src/smiles/cgan_{}.txt".format(i), 'r') as f:
            this_smiles = f.read().split('\n')[:-1]
            f.close()

        this_mols = [Chem.MolFromSmiles(this) for this in this_smiles].remove(None)
        
        if this_mols == None:
            continue
        
        for mol in this_mols:
            try:
                scores = utils.all_scores(mols=mol, data=data)
                file_to_write.write(f"Molecule:\n{mol}:\nScores:\n{scores}\n\n")
            except:
                pass
        
    file_to_write.close()