import matplotlib.pyplot as plt
import networkx as nx
import csv
import os
import logging
from typing import Optional
from datetime import datetime
from utils.logger import get_logger
logger = get_logger(__name__)

def plot_graph(G: nx.Graph, labels: Optional[dict] = None, title: str = "Graph", output_dir: str = "outputs", save_as: Optional[str] = None, method: Optional[str] = None) -> None:
    """
    Plots a graph using matplotlib and saves node labels to a CSV file and optionally an image.

    Args:
        G (nx.Graph): The graph to plot.
        labels (dict, optional): A dictionary of node labels. If None, will use 'label' attribute from G.
        title (str): Title for the plot.
        output_dir (str): Directory where CSV and image output are saved.
        save_as (str, optional): Base filename to save the plot image (e.g., 'plot.png').
        method (str, optional): The method name to include in the filename for tracking.
    """
    labels = nx.get_node_attributes(G, "label")
    '''
    if not labels:
        logging.warning("No labels found for nodes. Skipping plot.")
        return
    '''
    # Assign colors based on unique labels
    unique_labels = list(set(labels.values()))
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    node_colors = [label_to_color[labels[node]] for node in G.nodes()]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_suffix = f"_{method}" if method else ""

    # CSV output
    csv_filename = f"node_labels{method_suffix}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "Label"])
        for node, label in labels.items():
            writer.writerow([node, label])
    logger.info(f"Node labels saved to {csv_path}")

    # Plotting
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=500,
        font_color='black',
        with_labels=True if labels else False
    )
    plt.title(title)
    plt.tight_layout()

    if save_as:
        image_filename = f"{os.path.splitext(save_as)[0]}{method_suffix}_{timestamp}.png"
        image_path = os.path.join(output_dir, image_filename)
        plt.savefig(image_path)
        logger.info(f"Graph plot saved as {image_path}")
    else:
        plt.show()
