import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(max_seq_len, d_model):
    """
    Calculates the Sinusoidal Positional Encoding based on
    the formula in the Transformer architecture.
    """
    pos_enc = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            # Applying Sine to even indices
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            # Applying Cosine to odd indices
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i)) / d_model)))
    return pos_enc

seq_len = 50
d_model = 128

pe = get_positional_encoding(seq_len, d_model)

plt.figure(figsize=(12, 8))
plt.pcolormesh(pe, cmap='RdBu') # Red-Blue colormap like the slides
plt.xlabel('Embedding Dimension')
plt.ylabel('Token Position')
plt.colorbar(label='Encoding Value')
plt.title('Generated Sinusoidal Positional Encoding')
plt.savefig('positional_encoding.png')
print("Positional Encoding plot saved as positional_encoding.png")
plt.show()