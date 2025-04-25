import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("viterbinet_results.csv")

df['accuracy'] = df['accuracy'].apply(lambda x: float(str(x).replace("tensor(", "").replace(")", "")))

plt.figure(figsize=(10, 6))

for l_value in sorted(df['l'].unique()):
    subset = df[df['l'] == l_value]
    plt.plot(subset['snr_db'], subset['accuracy'], marker='o', label=f'l={l_value}')

plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs SNR for different window sizes (l)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()