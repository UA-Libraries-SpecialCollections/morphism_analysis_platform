# Standalone Morphism Analysis and Evidence Documentation Platform

Run with Python 3.10:

```bash
python morphism_analysis_platform.py
```

Or open a comparison file at launch:

```bash
python morphism_analysis_platform.py path/to/morphism_comparison.pkl
```

## Components

1. **Schema inspector** — shows payload kind/version, document/edge/match counts, structured-array fields, `match_diagnostics`, and `plot_cache` contents.
2. **Match query workbench** — filters retained matches by document, match type, Δ, PC1, Q, lexical divergence/overlap, acuity, and document embedding cosine; exports the current table.
3. **Plot-cache graph views** — restores compact graph views from `plot_cache`: count, lexical overlap, lexical divergence, peak acuity, acute candidates, lexical-Z/peak-acuity, document embedding cosine, and anchor-edge contribution.
4. **Candidate evidence browser** — browses cached top candidates or acuity-ranked fallback matches; exports Markdown/JSON evidence packets.
5. **Edge match 3D** — renders one selected match as a local 3D edge-pair diagram with four endpoint cluster centroids, source/destination displacement arrows, endpoint PC1 arrows, and color-coded re-embedded segment point clouds.

## Edge match 3D workflow

The new 3D view is query-driven:

1. Open `morphism_comparison.pkl`.
2. Open companion `document_delta_dict.pkl`.
3. Open companion `segments_by_doc.pkl`.
4. Run a query in the match query workbench.
5. Select a match row.
6. Click **Open selected in 3D**.
7. In the **Edge match 3D** tab, click **Render**.

The 3D view does not require a saved segment-embedding store. It re-embeds only the four selected endpoint clusters using SentenceTransformers, defaulting to `all-MiniLM-L6-v2`, which matches the current pipeline default. The projection is a local NumPy PCA over the selected segment clouds, stored centroids, and vector endpoints.

### What the 3D view draws

- translucent points: re-embedded segment point clouds for each selected endpoint cluster
- circles: source endpoint centroids
- triangles: destination endpoint centroids
- thick arrows: source→destination displacement vectors Δ for the two morphisms
- thin arrows: endpoint PC1 directions

The **canonicalize PC1 to edge flow** checkbox orients each source PC1 with the displacement vector and each destination PC1 against the incoming displacement direction. This follows the project’s zig-zag morphism convention and usually makes alignment comparisons easier to read.

## Dependencies

Install the requirements:

```bash
pip install -r requirements_morphism_analysis_platform.txt
```

For score browsing only, `numpy`, `matplotlib`, and optionally `pandas` are enough. For the Edge match 3D point clouds, `sentence-transformers` is required because the app re-embeds selected segment texts on demand.

## Large-file loading behavior

The app uses staged loading for large `morphism_comparison.pkl` files. Opening a file deserializes the pickle on a background thread and refreshes only schema/plot-cache metadata first. Match queries, candidate tables, graph rendering, and 3D edge rendering are run manually from their tabs so million-row comparisons do not freeze the Tkinter UI immediately after opening.

The data model caches row-aligned NumPy fields after load. This avoids repeated million-row array conversions while populating the match table and evidence browser.

After loading, use:

- **Run query** on the query tab to populate the match table.
- **Refresh candidates** on the evidence tab to populate candidate rows. The default candidate limit is 1,000; increase it only when needed.
- **Render** on the graph tab to draw the selected plot-cache view.
- **Open selected in 3D** on the query tab to inspect a selected match as a local edge-pair diagram.

## Optional companion artifacts

The platform can browse scores and candidates from `morphism_comparison.pkl` alone. Additional features require companion artifacts:

- `document_delta_dict.pkl` enables centroid, displacement, PC1, and Q geometry for the Edge match 3D tab.
- `segments_by_doc.pkl` enables evidence-browser cluster texts, lexical checks, and Edge match 3D re-embedded point clouds.

## Security

Pickle files can execute arbitrary code when loaded. Open only pickle files you created or that come from trusted collaborators.
