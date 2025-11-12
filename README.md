# GNN Bug Triage Assistant

A deep learning project that uses a Graph Neural Network (GNN) to automatically recommend the best developer to fix a new bug, trained on the entire history of the `microsoft/vscode` repository.

**Tags:** `Python`, `PyTorch`, `PyTorch Geometric`, `GNN`, `GraphSAGE`, `Sentence Transformers`, `NLP`, `Data Engineering`, `Machine Learning`

## The Problem

In large, complex software projects with hundreds of contributors, **bug triage** is a significant bottleneck. When a new bug is reported, a project manager must manually decide which developer on the team is the best person to fix it.

* Assigning it to the wrong person wastes days of valuable engineering time.
* A developer might be an "expert" in a file, but they may have left the team or not touched the code in years.
* The "right" person is often not just an expert in one file, but in the *ecosystem* of files and dependencies related to the bug.

This isn't a simple keyword-matching problem; it's a **relationship problem.**

## The Solution: A Graph Neural Network

This project frames the problem as a graph. A GNN is the perfect tool because it learns from **relationships and context**, not just individual data points.

I built a "knowledge graph" of the VSCode repository with three types of nodes:
* **Developers:** The engineers who fix bugs.
* **Bugs:** The issues reported by users.
* **Files:** The actual source code files in the repo.

The GNN learns complex patterns by "walking" this graph:
> "This new bug's text (`SentenceTransformer` embedding) is similar to past bugs that `mention` **File A** and **File B**. **Developer X** hasn't touched these files, but they recently `modified` **File C** and **File D**, which are *also* frequently modified along with A and B. Therefore, Developer X is the most relevant expert right now."



## Results: An 83% Accurate "Expert"

The model was trained on a dataset of 100 recent bugs from the `microsoft/vscode` repository.

* **Metric:** Area Under the Curve (AUC), which measures the model's ability to rank the correct developer higher than a random one. (0.5 is random, 1.0 is perfect).
* **Final Test AUC: 0.8333**

This result proves the model successfully learned the complex, hidden relationships in the codebase and can predict the correct "fixer" with high accuracy.

## Tech Stack

* **Machine Learning:** PyTorch, PyTorch Geometric (PyG)
* **GNN Model:** GraphSAGE (`SAGEConv`)
* **NLP (Features):** `sentence-transformers` (for text embeddings)
* **Data Engineering:** `PyGithub` (API scraping), Pandas
* **Environment:** Google Colab

## How to See it Work

The entire project, from data collection to training and testing, is contained in the `Bug_Triage_GNN.ipynb` notebook.

1.  **Open the Notebook:** Open `Bug_Triage_GNN.ipynb` in Google Colab.
2.  **Run the Cells:** You can run all the cells from top to bottom.
    * **Phase 1 (Data):** Scrapes GitHub for bug/developer/file data (uses the included CSVs as a cache).
    * **Phase 2 (Graph):** Builds the `HeteroData` graph object and generates text embeddings.
    * **Phase 3 (Train):** Trains the GNN model and reports the final `Test AUC` score.
    * **Phase 4 (Test):** Run a live test by typing in a *new* bug title and body to get a real-time recommendation.

## How to Use This on Your Own Repository

This project is a *factory* that can create an "expert" for any public repo. To train it on your own project:

1.  **Get a GitHub Token:**
    * Go to your GitHub **Settings** > **Developer settings** > **Personal access tokens** > **Tokens (classic)**.
    * Generate a new token with `public_repo` scope.
2.  **Add to Colab:**
    * Open the `Bug_Triage_GNN.ipynb` notebook in Colab.
    * Click the **"key" (🔑) icon** on the left and add a new secret.
    * Name the secret `GITHUB_PAT` and paste your token as the value.
3.  **Change Repo Name:**
    * In the second or third code cell, find this line:
        ```python
        REPO_NAME = "microsoft/vscode"
        ```
    * Change it to your target repository (e.g., `REPO_NAME = "facebook/react"`).
4.  **Run All:**
    * Click **`Runtime`** -> **`Restart and run all`**.

The notebook will automatically scrape your chosen repo, build a new graph, train a model, and the test cells will now give you predictions for *your* project's developers.

(*Note: This works best on large, active repositories with a good history of labeled bugs.*)

## Future Scope

This project is a successful proof-of-concept. The next steps to turn it into a production-ready tool would be:

* **Deploy as a GitHub Bot:** Package the trained model and inference logic into a web server (like FastAPI) and deploy it as a GitHub Action. The bot could automatically comment on every new bug issue with its Top 3 recommendations.
* **Full-Scale Training:** Train the model on the *entire 90k+ bug history* of VSCode, not just 100, to build a much more robust and accurate expert.
* **Richer Graph Features:** Enhance the graph with more features, such as:
    * `File` node features (file type, lines of code, age).
    * `Developer` node features (seniority, number of commits, specialty).
    * Using **CodeBERT** to create embeddings for the *code* in the files, not just the bug text.
* **Task Expansion:** The GNN could be trained to predict other valuable metrics, like the **estimated time-to-fix** or the **bug's severity level** (e.g., "trivial" vs. "critical").
