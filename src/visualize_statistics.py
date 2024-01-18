import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(statistics, path_to_save):
    df = pd.read_json(statistics).transpose()

    plt.figure(figsize=(10, 6))

    hot_colors = plt.cm.hot(np.linspace(0.25, 0.75, len(df.columns)))
    cold_colors = plt.cm.cool(np.linspace(0.25, 0.75, len(df.columns)))

    for i, column in enumerate(df.columns):
        if "G_" in column:
            plt.plot(df.index, df[column], label=column, color=hot_colors[i])
        else:
            plt.plot(df.index, df[column], label=column, color=cold_colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Epochs')
    plt.legend()
    
    plt.savefig(f'{path_to_save}/plot.png')