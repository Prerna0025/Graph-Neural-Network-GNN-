import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Dict, List, Tuple
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def get_timestamped_filename(base_name: str, ext: str = "csv") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/{base_name}_{timestamp}.{ext}"


def encode_labels(final_labels: Dict[int, str]) -> np.ndarray:
    label_values = list(final_labels.values())
    encoder = LabelEncoder()
    label_ids = encoder.fit_transform(label_values)
    logger.info(f"Encoded label indices: {label_ids}")

    # Save label IDs
    pd.DataFrame(label_ids, columns=["LabelID"]).to_csv(get_timestamped_filename("label_ids"), index=False)
    return label_ids


def encode_onehot_labels(final_labels: Dict[int, str]) -> np.ndarray:
    label_values = list(final_labels.values())
    encoder = OneHotEncoder(sparse_output=False)
    reshaped = np.array(label_values).reshape(-1, 1)
    onehot_labels = encoder.fit_transform(reshaped)
    logger.info(f"One-hot encoded labels shape: {onehot_labels.shape}")

    # Save one-hot encodings
    pd.DataFrame(onehot_labels).to_csv(get_timestamped_filename("onehot_labels"), index=False)
    return onehot_labels


def hex_to_normalized_values(final_labels: Dict[int, str]) -> None:
    normalized_data = []
    for key, value in final_labels.items():
        int_val = int(value, 16)
        normalized = int_val / (2 ** 128 - 1)
        normalized_data.append((key, normalized))
        logger.debug(f"Node {key}: Normalized Value: {normalized:.6f}")

    df = pd.DataFrame(normalized_data, columns=["Node", "Normalized"])
    df.to_csv(get_timestamped_filename("normalized_values"), index=False)
    logger.info("Normalized values saved")


class GraphEmbeddingModel(nn.Module):
    def __init__(self, num_wl_labels: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_wl_labels, embedding_dim)

    def forward(self, wl_idx_u: torch.Tensor, wl_idx_v: torch.Tensor) -> torch.Tensor:
        emb_u = self.embedding(wl_idx_u)
        emb_v = self.embedding(wl_idx_v)
        score = torch.sum(emb_u * emb_v, dim=1)
        return torch.sigmoid(score)


def train_graph_embeddings(
    final_labels: Dict[int, str],
    edge_list: List[Tuple[int, int]],
    embedding_dim: int = 32,
    epochs: int = 50,
    save_path: str = None
) -> None:
    os.makedirs("outputs", exist_ok=True)

    wl_labels = list(final_labels.values())
    label_encoder = LabelEncoder()
    wl_label_ids = label_encoder.fit_transform(wl_labels)

    wl_idx_u = torch.tensor([wl_label_ids[u] for u, v in edge_list])
    wl_idx_v = torch.tensor([wl_label_ids[v] for u, v in edge_list])

    logger.info(f"WL idx tensors created with {len(wl_idx_u)} edges")

    nodes = list(set([node for edge in edge_list for node in edge]))
    neg_edges = []
    while len(neg_edges) < len(edge_list):
        u, v = random.sample(nodes, 2)
        if (u, v) not in edge_list and (v, u) not in edge_list:
            neg_edges.append((u, v))

    model = GraphEmbeddingModel(num_wl_labels=len(label_encoder.classes_), embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        u_pos = torch.tensor([wl_label_ids[u] for u, v in edge_list])
        v_pos = torch.tensor([wl_label_ids[v] for u, v in edge_list])
        label_pos = torch.ones(len(edge_list))

        u_neg = torch.tensor([wl_label_ids[u] for u, v in neg_edges])
        v_neg = torch.tensor([wl_label_ids[v] for u, v in neg_edges])
        label_neg = torch.zeros(len(neg_edges))

        u_all = torch.cat([u_pos, u_neg])
        v_all = torch.cat([v_pos, v_neg])
        labels = torch.cat([label_pos, label_neg])

        preds = model(u_all, v_all)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    final_embeddings = model.embedding.weight.data.cpu().numpy()
    df = pd.DataFrame(final_embeddings)
    output_path = save_path or get_timestamped_filename("wl_dense_embeddings")
    df.to_csv(output_path, index=False)
    logger.info(f"Embeddings saved to {output_path}")
