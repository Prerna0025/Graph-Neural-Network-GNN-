# WL Graph Embedding Project

A modular and configurable implementation of the Weisfeiler-Lehman (WL) 1-dimensional algorithm for graph node labeling, embedding generation, and visualization.

This project is built using PyTorch, NetworkX, PyTorch Geometric, and provides support for:
- WL-based label computation
- Dense embedding generation using self-supervised training
- Label encodings (One-Hot, Label ID, Hex Normalization)
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
encoding: "dense"   # Options: "dense", "label", "onehot", "hex"
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
