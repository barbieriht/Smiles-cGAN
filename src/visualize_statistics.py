import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(statistics, path_to_save):
    df = pd.read_json(statistics).transpose()

    # Split columns into "D " and "G " groups
    d_columns = [column for column in df.columns if "D " in column]
    g_columns = [column for column in df.columns if "G " in column]

    plt.figure(figsize=(14, 6))

    # Plot "D " columns in the first subplot
    plt.subplot(2, 1, 1)
    for i, column in enumerate(d_columns):
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('D Losses Over Epochs')
    plt.legend()

    # Plot "G " columns in the second subplot
    plt.subplot(2, 1, 2)
    for i, column in enumerate(g_columns):
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('G Losses Over Epochs')
    plt.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{path_to_save}/plot.png')