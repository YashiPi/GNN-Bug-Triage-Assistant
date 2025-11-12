# GNN Bug Triage Assistant

A deep learning project that uses a Graph Neural Network (GNN) to automatically recommend the best developer to fix a new bug, trained on the entire history of the `microsoft/vscode` repository.

**Tags:** `Python`, `PyTorch`, `PyTorch Geometric`, `GNN`, `GraphSAGE`, `Sentence Transformers`, `NLP`, `Data Engineering`, `Machine Learning`

## 🚀 The Problem

In large, complex software projects with hundreds of contributors, **bug triage** is a significant bottleneck. When a new bug is reported, a project manager must manually decide which developer on the team is the best person to fix it.

* Assigning it to the wrong person wastes days of valuable engineering time.
* A developer might be an "expert" in a file, but they may have left the team or not touched the code in years.
* The "right" person is often not just an expert in one file, but in the *ecosystem* of files and dependencies related to the bug.

This isn't a simple keyword-matching problem; it's a **relationship problem.**

## 💡 The Solution: A Graph Neural Network

This project frames the problem as a graph. A GNN is the perfect tool because it learns from **relationships and context**, not just individual data points.

I built a "knowledge graph" of the VSCode repository with three types of nodes:
* **Developers:** The engineers who fix bugs.
* **Bugs:** The issues reported by users.
* **Files:** The actual source code files in the repo.

The GNN learns complex patterns by "walking" this graph:
> "This new bug's text (`SentenceTransformer` embedding) is similar to past bugs that `mention` **File A** and **File B**. **Developer X** hasn't touched these files, but they recently `modified` **File C** and **File D**, which are *also* frequently modified along with A and B. Therefore, Developer X is the most relevant expert right now."



## 📈 Results: An 83% Accurate "Expert"

The model was trained on a dataset of 100 recent bugs from the `microsoft/vscode` repository.

* **Metric:** Area Under the Curve (AUC), which measures the model's ability to rank the correct developer higher than a random one. (0.5 is random, 1.0 is perfect).
* **Final Test AUC: 0.8333**

This result proves the model successfully learned the complex, hidden relationships in the codebase and can predict the correct "fixer" with high accuracy.

## 🛠️ Tech Stack

* **Machine Learning:** PyTorch, PyTorch Geometric (PyG)
* **GNN Model:** GraphSAGE (`SAGEConv`)
* **NLP (Features):** `sentence-transformers` (for text embeddings)
* **Data Engineering:** `PyGithub` (API scraping), Pandas
* **Environment:** Google Colab

## 🔬 How to See it Work

The entire project, from data collection to training and testing, is contained in the `Bug_Triage_GNN.ipynb` notebook.

1.  **Open the Notebook:** Open `Bug_Triage_GNN.ipynb` in Google Colab.
2.  **Run the Cells:** You can run all the cells from top to bottom.
    * **Phase 1 (Data):** Scrapes GitHub for bug/developer/file data (uses the included CSVs as a cache).
    * **Phase 2 (Graph):** Builds the `HeteroData` graph object and generates text embeddings.
    * **Phase 3 (Train):** Trains the GNN model and reports the final `Test AUC` score.
    * **Phase 4 (Test):** Run a live test by typing in a *new* bug title and body to get a real-time recommendation.
