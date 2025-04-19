# WL Graph Embedding Project

A modular and configurable implementation of the Weisfeiler-Lehman (WL) 1-dimensional algorithm for graph node labeling, embedding generation, and visualization.
## 🧩 What is the Weisfeiler-Lehman (1-WL) Algorithm?

The 1-WL algorithm (also known as color refinement) is a graph isomorphism test used to iteratively update node labels based on their neighbors. It operates as follows:

1. Assign each node an initial label (e.g., degree or a default label).
2. At each iteration:
   - Concatenate the label of a node with sorted labels of its neighbors.
   - Apply a hash function to generate a new label.
3. Repeat until labels converge or max iterations reached.

This process encodes **structural similarity** into node labels — nodes with similar neighborhoods will get similar hash values.

---

## 📌 How 1-WL Labels are Used

After computing 1-WL labels for each node, we use them for **embedding generation** in multiple ways:

- **Label Encoding**: Convert hash strings to integers.
- **One-Hot Encoding**: Convert label IDs to sparse binary vectors.
- **Hex Normalization**: Normalize hash values to float features between 0 and 1.
- **Dense Embedding (Self-Supervised)**:
    - Treat label pairs from edges as training examples.
    - Train an `nn.Embedding` model using positive (real) and negative (random) node pairs.
    - Embeddings are learned such that connected nodes are close in vector space.

---

## 🔮 How to Use These Embeddings

These embeddings can be:
- Used as **additional node features** in Graph Neural Networks (GNNs).
- Fed into downstream ML models for node classification or clustering.
- Visualized to understand graph structure.
- Combined with other features (e.g., node degree, community info) to boost performance.

This project is built using PyTorch, NetworkX, PyTorch Geometric, and provides support for:
- WL-based label computation
- Dense embedding generation using self-supervised training
- Label encodings (One-Hot, Label ID, Hex Normalization, Dense vector)
- Configurable YAML setup
- Modular structure for experimentation
- Graph visualization and CSV export

---

## 📁 Project Structure

```
wl_project/
│
├── configs/                    # YAML config files
│   └── default.yaml
│
├── data/                       # Graph datasets (TUDataset, etc.)
│
├── encoders/                  # Encoding and training modules
│   ├── wl_encoder.py
│   
│
├── algorithms/
│   ├── WL.py                   # Weisfeiler-Lehman algorithm
│   └── Graph_plots.py          # Visualization and CSV export
│
├── utils/
│   └── logger.py               # Logger setup
│
├── wl_main.py                  # Main script (with argparse or YAML support)
└── requirements.txt
```

---

## ⚙️ Configuration

The project is controlled via `configs/default.yaml`:

```yaml
dataset: "PROTEINS"
graph_index: 0
hash_method: "md5"
max_itr: 10
all_iterations: false
embedding_dim: 32
epochs: 50
encoding: "NormalizedValue"   # Options: "onehot", "IntegerMapping", "NormalizedValue", "DenseVector"
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main script (YAML mode)

```bash
python wl_main.py
```

Or to use argparse (if implemented):

```bash
python wl_main.py --dataset PROTEINS --graph_index 0 --hash_method sha1 --encoding dense
```

---

## 🧠 Features

- 🔄 Supports multiple hash methods: `md5`, `sha1`, `sha256`, `sha512`
- 🧪 Dense self-supervised training via `nn.Embedding`
- 🧬 Hex-to-normalized float vector mapping
- 📊 Visualization with `matplotlib` and color-coded WL labels
- 📝 Saves node labels, one-hot encodings, and final dense embeddings as CSV

---

## 🖼️ Example Output

Graph visualization is saved in `/outputs/graph_<method>_<timestamp>.png`.

Embeddings are saved as:

- `wl_dense_embeddings_<timestamp>.csv`
- `onehot_labels_<timestamp>.csv`
- `label_ids_<timestamp>.csv`

---

## 🔧 Requirements

- Python 3.8+
- torch
- torch-geometric
- networkx
- matplotlib
- pandas
- scikit-learn
- omegaconf
- tqdm

(installable via `requirements.txt`)

---

## ✨ Credits

Developed as part of an independent research project on Graph Neural Networks and Embedding Learning.

---

## 📄 License

MIT License. See `LICENSE` file.
