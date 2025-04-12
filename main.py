import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from viterbinet import ViterbiNet
from data_generator import generate_isi_data
from mixture_model import GMMMarginal
from viterbi_decoder import decode_with_viterbinet

l = 3
num_states = 2 ** l
hidden_dim = 64
num_samples = 100000
snr_db = 8

Y, X_seq = generate_isi_data(num_samples, l , snr_db)
X_seq = X_seq[:len(Y)]
X_labels = [int(''.join(str(int((s+1)//2)) for s in x), 2) for x in X_seq]

combined = list(zip(Y, X_labels))
train_data, val_data = train_test_split(combined, test_size=0.2, random_state=42)
Y_train, X_train = zip(*train_data)
Y_val, X_val = zip(*val_data)

Y_train_tensor = torch.tensor(Y_train).float().unsqueeze(1)
X_train_tensor = torch.tensor(X_train, dtype=torch.long)

Y_val_tensor = torch.tensor(Y_val).float().unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)

model = ViterbiNet(input_dim=1, hidden_dim=hidden_dim, num_states=num_states)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.NLLLoss()

best_val_acc = 0
print("Training ViterbiNet...")
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    log_probs = model(Y_train_tensor)
    loss = loss_fn(log_probs, X_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = log_probs.argmax(dim=1)
        train_acc = (train_preds == X_train_tensor).float().mean().item()

        val_log_probs = model(Y_val_tensor)
        val_loss = loss_fn(val_log_probs, X_val_tensor)
        val_preds = val_log_probs.argmax(dim=1)
        val_acc = (val_preds == X_val_tensor).float().mean().item()

    print(f"Epoch {epoch+1}/{10} - Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

print("Fitting GMM on observed outputs...")
gmm = GMMMarginal(num_components=5)
gmm.fit(Y_train_tensor)

print("Running Viterbi decoding...")
decoded = decode_with_viterbinet(model, Y_val_tensor, gmm, l)\

print("\nDecoded symbols (first 20):")
print(decoded[:20])

true_bits = [(int((x[0] + 1) // 2)) for x in [X_seq[i] for i in range(len(Y)) if Y[i] in Y_val]]
symbol_map = {0: -1, 1: 1}
true_symbols = [symbol_map[b] for b in true_bits[:len(decoded)]]

print("True symbols (first 20):")
print(true_symbols[:20])

accuracy = sum([a == b for a, b in zip(decoded, true_symbols)]) / len(decoded)
print(f"\nSymbol accuracy: {accuracy * 100:.2f}%")