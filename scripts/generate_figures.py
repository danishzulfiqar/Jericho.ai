"""Generate README figures from the committed FAISS vector stores.

Requires the same environment as app.py (langchain 0.0.154, faiss-cpu, openai 0.28)
plus scikit-learn and matplotlib. No API calls are made.

    python scripts/generate_figures.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Assets"
OUT.mkdir(exist_ok=True)

INK, MUTED, ACCENT, COPPER, SAGE, LINE = "#202c3a", "#536171", "#305b8a", "#b47b40", "#5f8a6e", "#dce1e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
COURSES = {
    "Islamiat": ("Islamic Studies (HUM112)", ACCENT),
    "Pakistan Studies": ("Pakistan Studies (HUM111)", COPPER),
    "Report Writting Lecture 1-15": ("Report Writing (Lectures 1–15)", SAGE),
}


def load(name):
    with open(ROOT / "VectorStore" / f"{name}.pkl", "rb") as f:
        vs = pickle.load(f)
    vecs = vs.index.reconstruct_n(0, vs.index.ntotal)
    texts = [vs.docstore.search(vs.index_to_docstore_id[i]).page_content for i in range(vs.index.ntotal)]
    return vecs, texts


def main():
    vecs, labels, lengths = [], [], {}
    for name in COURSES:
        v, t = load(name)
        vecs.append(v)
        labels += [name] * len(v)
        lengths[name] = [len(x) for x in t]
    X = np.vstack(vecs)
    y = np.array(labels)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

    knn_acc = cross_val_score(KNeighborsClassifier(5, metric="cosine"), Xn, y, cv=5).mean()
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -1)
    nn_sim = sims.max(axis=1)

    # 1. t-SNE map of the embedding space
    emb = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=42).fit_transform(Xn)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for name, (label, color) in COURSES.items():
        m = y == name
        ax.scatter(emb[m, 0], emb[m, 1], s=16, color=color, alpha=0.78, edgecolor="white", linewidth=0.3,
                   label=f"{label}  ·  {m.sum()} chunks")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("Course material in embedding space (t-SNE of 1,536-d ada-002 vectors)", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
    fig.savefig(OUT / "embedding_map.png")
    plt.close(fig)

    # 2. Knowledge-base composition: chunk counts and chunk lengths
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    names = list(COURSES)
    counts = [int((y == n).sum()) for n in names]
    short = [COURSES[n][0].split(" (")[0] for n in names]
    a1.barh(short, counts, color=[COURSES[n][1] for n in names], height=0.6)
    for i, c in enumerate(counts):
        a1.text(c + 5, i, str(c), va="center", color=MUTED, fontsize=9)
    a1.set_xlim(0, max(counts) * 1.2)
    a1.invert_yaxis()
    a1.set_xlabel("Indexed chunks")
    a1.set_title("Chunks per course", loc="left")
    bins = np.linspace(0, 1000, 26)
    for n in names:
        a2.hist(lengths[n], bins=bins, color=COURSES[n][1], alpha=0.65, label=COURSES[n][0].split(" (")[0])
    a2.axvline(1000, color=INK, ls="--", lw=1)
    a2.text(985, a2.get_ylim()[1] * 0.92, "chunk_size = 1000", ha="right", fontsize=8, color=INK)
    a2.set_xlabel("Chunk length (characters)")
    a2.set_ylabel("Chunks")
    a2.set_title("Chunk length distribution (overlap = 200)", loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "knowledge_base.png")
    plt.close(fig)

    stats = {
        "total_chunks": int(len(y)), "embedding_dim": int(X.shape[1]),
        "chunks_per_course": dict(zip(names, counts)),
        "mean_chunk_chars": {n: round(float(np.mean(lengths[n])), 1) for n in names},
        "course_separability_knn5_cv": round(float(knn_acc), 4),
        "median_nearest_neighbour_cosine": round(float(np.median(nn_sim)), 4),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
