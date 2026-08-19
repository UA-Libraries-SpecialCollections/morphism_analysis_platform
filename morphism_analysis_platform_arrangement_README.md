# Morphism Analysis Platform — Arrangement Experiment Update

This version adds a **6. Arrangement experiment** tab for a simple mixed-collection experiment over shared morphism-shape category membership.

## Required input

Load a companion `document_delta_dict.pkl` with **File → Open companion document_delta_dict.pkl...**.

The tab builds:

- morphism-shape records from intra-document directed cluster edges
- shape categories using k-means over zig-zag shape features
- a document × shape-category membership matrix
- collection labels from doc ID prefix, regex, metadata CSV, or `Unknown`
- document similarity over shape-membership rows
- shape-category co-occurrence from shared document membership support

## New views

The Arrangement experiment tab includes:

- Summary
- Shape atlas / representative edges
- Membership heatmap and similar-doc table
- Document graph by shared shape membership
- Shape-category co-occurrence graph
- Collection summary / dominant-shape bars

## Experiment artifact

Use **Save experiment PKL** to write:

`morphism_arrangement_experiment.pkl`

Use **Export tables** to export CSV tables:

- `shape_summary.csv`
- `representative_edges.csv`
- `collection_summary.csv`
- `doc_shape_membership.csv`

## Default experiment settings

- `shape_k = 48`
- `include_length = true`
- `include_v_bin = false`
- `dir_weight_beta = 1.0`
- `max_edges_per_doc = 2000`
- `min_weight = 0.0`
- `weight_mode = weighted`
- `normalize_membership = true`
- `similarity = cosine`
- `doc_top_k = 5`
- `support ≥ 0.03`

