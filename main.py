import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from viterbinet import ViterbiNet
from data_generator import generate_isi_data
from mixture_model import GMMMarginal
from viterbi_decoder import decode_with_viterbinet
from kde_model import KDEMarginal

l = 3
num_states = 2 ** l
hidden_dim = 20
num_samples = 100000
snr_db = 16
batch_size = 256
window_size = 3
num_epochs = 20
lr = 0.001
window_size = 3
val_ratio = 0.2
test_ratio = 0.1

torch.manual_seed(42)
np.random.seed(42)

Y, X_seq = generate_isi_data(num_samples, l , snr_db)

X_seq = X_seq[:len(Y)]
X_labels = [int(''.join(str(int((s+1)//2)) for s in x), 2) for x in X_seq]

combined = list(zip(Y, X_labels))
train_data, val_data = train_test_split(combined, test_size=0.2, random_state=42)
Y_train, X_train = zip(*train_data)
Y_val, X_val = zip(*val_data)

Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
X_train_tensor = torch.tensor(X_train, dtype=torch.long)

Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)

train_dataset = TensorDataset(Y_train_tensor, X_train_tensor)
val_dataset = TensorDataset(Y_val_tensor, X_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = ViterbiNet(input_dim=1, hidden_dim=hidden_dim, num_states=num_states, window_size=window_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
loss_fn = torch.nn.CrossEntropyLoss()

print("Training ViterbiNet...")

"""
for epoch in range(100):
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
"""
# train_losses = []
# val_losses = []
# train_accs = []
# val_accs = []
best_val_acc = 0
patience = 15
patience_count = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, targets in tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}"):
    #     if window_size > 1:
    #         batch_size = inputs.size(0)
    #         windowed_inputs = torch.zeros(batch_size, window_size)

    #         half_window = window_size // 2
    #         for i in range(batch_size):
    #             for w in range(window_size):
    #                 idx = min(max(0, i - half_window + w), batch_size - 1)
    #                 windowed_inputs[i, w] = inputs[idx]
            
    #         outputs = model(windowed_inputs)
    #     else: 
    #         outputs = model(inputs)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(dim=1)
        train_correct += (pred == targets).sum().item()
        train_total += targets.size(0)

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / train_total
    # train_losses.append(train_loss)
    # train_accs.append(train_accs)


    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # if window_size > 1:
            #     batch_size = inputs.size(0)
            #     windowed_inputs = torch.zeros(batch_size, window_size)

            #     half_window = window_size // 2
            #     for i in range(batch_size):
            #         for w in range(window_size):
            #             idx = min(max(0, i - half_window + w), batch_size - 1)
            #             windowed_inputs[i, w] = inputs[idx]
                
            #     outputs = model(windowed_inputs)
            # else: 
            #     outputs = model(inputs)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            val_correct += (pred == targets).sum().item()
            val_total += targets.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / val_total
    # val_losses.append(val_loss)
    # val_accs.append(val_acc)
    
    # scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_count = 0

        torch.save(model.state_dict(), "best_viterbinet.pt")
        print(f"New best model validation accuracy: {val_acc:.4f}")
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"Early stop after {patience} epochs w/o improvement")
            break

model.load_state_dict(torch.load("best_viterbinet.pt"))

print("Fitting KDE on observed outputs...")
# gmm = GMMMarginal(num_components=8)
# gmm.fit(Y_train_tensor) 
kde = KDEMarginal(bandwidth=0.3)
kde.fit(Y_train_tensor)

print("Running Viterbi decoding...")
# decoded = decode_with_viterbinet(model, Y_val_tensor, gmm, l)
decoded = decode_with_viterbinet(model, Y_val_tensor, kde, l)

true_bits = [(int((x[0] + 1) // 2)) for x in [X_seq[i] for i in range(len(Y)) if Y[i] in Y_val]]
symbol_map = {0: -1, 1: 1}
true_symbols = [symbol_map[b] for b in true_bits[:len(decoded)]]

print("\nDecoded symbols (first 20):")
print(decoded[:20])

print("True symbols (first 20):")
print(true_symbols[:20])

accuracy = sum([a == b for a, b in zip(decoded, true_symbols)]) / len(decoded)
print(f"\nSymbol accuracy: {accuracy * 100:.2f}%")

# min_len = min(len(decoded), len(true_symbols))
# decoded = decoded[:min_len]
# true_symbols = true_symbols[:min_len]

# accuracy = np.mean(np.array(decoded) == np.array(true_symbols))
# print(f"Symbol accuracy: {accuracy * 100:.2f}%")

print("\nConfusion matrix:")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_symbols, decoded)
print(cm)