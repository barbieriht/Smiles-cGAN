import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(statistics, batch_per_epoca, learning_rate):
    df = pd.read_json(statistics).transpose()

    plt.figure(figsize=(10, 6))

    # Plot the first three columns with hot colors
    hot_colors = plt.cm.hot(np.linspace(0.25, 0.75, 3))
    for i, column in enumerate(df.columns[:3]):
        plt.plot(df.index, df[column], label=column, color=hot_colors[i])

    # Plot the rest of the columns with cold colors
    cold_colors = plt.cm.cool(np.linspace(0.25, 0.75, len(df.columns) - 3))
    for i, column in enumerate(df.columns[3:]):
        plt.plot(df.index, df[column], label=column, color=cold_colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Epochs')
    plt.legend()
    
    plt.savefig(f'./generated_files/lr{learning_rate}_bpe{batch_per_epoca}/plot.png')