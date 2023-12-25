import pandas as pd
import matplotlib.pyplot as plt

def plot(statistics, batch_per_epoca, learning_rate):
    df = pd.read_json(statistics).transpose()

    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Epochs')
    plt.legend()
    
    plt.savefig(f'bpe{batch_per_epoca}_lr{learning_rate}.png')