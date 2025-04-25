import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("viterbinet_results.csv")

df['accuracy'] = df['accuracy'].apply(lambda x: float(str(x).replace("tensor(", "").replace(")", "")))

df['ber'] = 1 - df['accuracy']

plt.figure(figsize=(10, 6))

for l_value in sorted(df['l'].unique()):
    subset = df[df['l'] == l_value]
    plt.plot(subset['snr_db'], subset['ber'], marker='o', label=f'l={l_value}')

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Bit Error Rate vs SNR for different window sizes (l)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()