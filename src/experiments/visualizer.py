import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import load_config
from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.models.automata_model import ProbabilisticAutomata


config = load_config()
AUTOMATA_CONFIG = config["automata"]

AUTOMATA_SEED = AUTOMATA_CONFIG["seeds"][0]
ALPHABET_SIZE = AUTOMATA_CONFIG["alphabet_size"]
WINDOW_SIZE = AUTOMATA_CONFIG["window_size"]


def plot_automata_diagnostics(y_true, y_pred, y_scores, dataset_name):
    """
    4. GÖREV İSTERİ: Olasılıksal Otomata için Confusion Matrix ve PR Curve 
    grafiklerini canlı test verileriyle üretir ve kaydeder.
    """
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    dataset_clean = dataset_name.lower().replace(" ", "_")

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Automata Confusion Matrix - {dataset_name}', fontsize=11, fontweight='bold', pad=10)
    plt.grid(False)
    
    cm_path = os.path.join(output_dir, f"{dataset_clean}_automata_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> {dataset_clean}_automata_confusion_matrix.png başarıyla üretildi.")

    plt.figure(figsize=(7, 5))
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='purple', lw=2, label=f'Automata (AUC = {pr_auc:.4f})')
    
    baseline = sum(y_true) / len(y_true) if len(y_true) > 0 else 0
    plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.2f})')
    
    plt.xlabel('Recall (Duyarlılık)', fontsize=10)
    plt.ylabel('Precision (Kesinlik)', fontsize=10)
    plt.title(f'Automata Precision-Recall Curve - {dataset_name}', fontsize=11, fontweight='bold', pad=10)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    pr_path = os.path.join(output_dir, f"{dataset_clean}_automata_pr_curve.png")
    plt.savefig(pr_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> {dataset_clean}_automata_pr_curve.png başarıyla üretildi.")


def plot_parameter_sensitivity_heatmap(csv_path="results/outputs/automata_param_sensitivity_metrics.csv"):
    """3. MADDE İSTERİ: parameter_sensitivity_heatmap.png görselini üretir."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    pivot_df = df.pivot_table(
        index='window_size',
        columns='alphabet_size',
        values='f1_score',
        aggfunc='mean'
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f", cbar_kws={'label': 'F1-Score'})
    plt.title("Parametre Duyarlılık Analizi (F1-Score Heatmap)")
    plt.xlabel("Alfabe Boyutu (Alphabet Size)")
    plt.ylabel("Pencere Boyutu (Window Size)")
    plt.savefig("results/figures/parameter_sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("-> parameter_sensitivity_heatmap.png başarıyla üretildi.")

def plot_parameter_complexity_analysis(csv_path="results/outputs/automata_param_sensitivity_metrics.csv"):
    """3. MADDE İSTERİ: parameter_complexity_analysis.png görselini üretir."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = 'tab:red'
    ax1.set_xlabel('Parametre Kombinasyon İndeksi')
    ax1.set_ylabel('Durum (State) Sayısı', color=color)
    ax1.plot(df.index, df['num_states'], color=color, marker='o', label='State Sayısı')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Geçiş Yoğunluğu (Transition Density)', color=color)
    ax2.plot(df.index, df['transition_density'], color=color, marker='s', linestyle='--', label='Geçiş Yoğunluğu')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Parametre Değişiminin Otomata Kompleksliğine Etkisi")
    fig.tight_layout()
    plt.savefig("results/figures/parameter_complexity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("-> parameter_complexity_analysis.png başarıyla üretildi.")


def generate_live_automata_and_plots():
    print("\n--- FREKANSA DAYALI GERÇEK MODEL GÖRSELLEŞTİRMESİ BAŞLATILIYOR ---")
    np.random.seed(AUTOMATA_SEED)

    if not os.path.exists("data/processed/batadal_X_train_pc1.csv"):
        return

    X_train = pd.read_csv("data/processed/batadal_X_train_pc1.csv").values.flatten()

    transformer = SaxPaaTransformer(alphabet_size=ALPHABET_SIZE)
    train_patterns = transformer.transform(X_train, window_size=WINDOW_SIZE)
    
    model = ProbabilisticAutomata(smoothing=True)
    model.fit(train_patterns)
    
    os.makedirs("results/figures", exist_ok=True)

    sorted_states_by_freq = sorted(model.total_exits.items(), key=lambda x: x[1], reverse=True)
    top_states = [state for state, freq in sorted_states_by_freq[:10]]
    
    matrix = np.zeros((len(top_states), len(top_states)))
    for i, s1 in enumerate(top_states):
        for j, s2 in enumerate(top_states):
            matrix[i, j] = model.get_transition_probability(s1, s2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, xticklabels=top_states, yticklabels=top_states, cmap="Blues", fmt=".2f")
    plt.title("En Yoğun Durumların Geçiş Olasılıkları Matrisi")
    plt.savefig("results/figures/transition_probability_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("-> transition_probability_heatmap.png başarıyla üretildi.")

    G = nx.DiGraph()
    for src in top_states:
        for tgt in top_states:
            if model.transitions[src][tgt] > 0:
                prob = model.get_transition_probability(src, tgt)
                if prob > 0.15:
                    G.add_edge(src, tgt, weight=round(prob, 2))

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.5, seed=AUTOMATA_SEED)
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color="lightgreen")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.axis("off")
    plt.savefig("results/figures/automata_state_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("-> automata_state_diagram.png başarıyla üretildi.")
    
    plot_parameter_sensitivity_heatmap()
    plot_parameter_complexity_analysis()


if __name__ == "__main__":
    generate_live_automata_and_plots()