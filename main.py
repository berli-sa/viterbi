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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv
import os

output_file = "viterbinet_results.csv"

if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["l", "snr_db", "accuracy"])

# l = 3
hidden_dim = 20
num_samples = 100000
# snr_db = -4
batch_size = 256
window_size = 3
num_epochs = 10
lr = 0.001
val_ratio = 0.2
test_ratio = 0.1

torch.manual_seed(42)
np.random.seed(42)

def run_one(l, snr_db):
    num_states = 2 ** l
    Y, X_seq, S = generate_isi_data(num_samples, l , snr_db)
    # print("X_seq: ", len(X_seq))
    # print("Y: ", len(Y))
    # print("S:", len(S))

    X_labels = [int(''.join(str(int((s+1)//2)) for s in x), 2) for x in X_seq]

    Y_val = Y[:int(val_ratio * num_samples)]
    X_val = X_labels[:int(val_ratio * num_samples)]
    S_val = S[:int(val_ratio * num_samples)]
    # print("S_val:", len(S_val))

    Y_train = Y[int(val_ratio * num_samples):]
    X_train = X_labels[int(val_ratio * num_samples):]

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
    loss_fn = torch.nn.NLLLoss()

    print("Training ViterbiNet...")

    best_val_acc = 0
    patience = 15
    patience_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}"):
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

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

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

    print("Fitting KDE on observed outputs...")
    # gmm = GMMMarginal(num_components=10)
    # gmm.fit(Y_train_tensor) 
    kde = KDEMarginal(bandwidth=0.03)
    kde.fit(Y_train_tensor)

    print("Running Viterbi decoding...")
    # decoded, states = decode_with_viterbinet(model, Y_val_tensor, gmm, l)
    decoded, states = decode_with_viterbinet(model, Y_val_tensor, X_val_tensor, kde, l)
    print("states", len(states), states[:20])
    print("x_val", len(X_val), X_val[:20])

    accuracy = sum([a == b for a, b in zip(X_val,states)]) / len(states)
    print(f"\nState accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion matrix:", l, snr_db)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(X_val[:-1], states)
    print(cm)
    return accuracy

l_values = range(2, 10)         # l from 2 to 10
snr_values = range(-6, 11)      # SNR from -6 to 10

results = {}  # Dictionary to store accuracies for each l

for l in l_values:
    accuracies = []
    for snr in snr_values:
        print(f"Running pipeline with l={l}, SNR={snr}")
        acc = run_one(l, snr)
        accuracies.append(acc)
        with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([l, snr, acc])
    results[l] = accuracies
