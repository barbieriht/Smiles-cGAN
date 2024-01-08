import random

if __name__ == "__main__":
    file_name = 'chebi_selected_smiles.txt'
    n_subsets = 10

    with open(file_name, "r") as f:
        smiles = f.readlines()
        f.close()
        
    random.shuffle(smiles)

    smiles = smiles[:int(len(smiles)/n_subsets)+1]

    with open(f'src/chebi_selected_smiles_1of{n_subsets}_subset.txt', 'w') as f:
        f.write(''.join(smiles))
        f.close()