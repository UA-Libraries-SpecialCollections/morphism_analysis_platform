# Morphism Analysis Platform — Arrangement Experiment and Shape Bin Field

This README documents the arrangement-analysis component of `morphism_analysis_platform.py`. The Arrangement experiment is the collection-scale layer of the platform: it represents documents by the morphism-shape categories expressed by their internal directed cluster edges, then compares those document profiles to collection labels, document graphs, ROC baselines, and shape-bin field diagnostics.

## Conceptual purpose

The Arrangement experiment asks:

> When documents are represented by the distribution of morphism-action categories inside them, do they group according to collection type, document form, or other structural patterns?

This is different from raw document embedding search. It does not represent a document as one embedding vector alone. It represents a document as a profile over learned **shape bins**: categories of directed semantic-transition form.

## Required input

Load a companion `document_delta_dict.pkl` using:

```text
File → Open companion document_delta_dict.pkl...
```

This file provides the CDM geometry needed to extract all directed cluster-to-cluster morphism edges:

- cluster centroids
- displacement vectors Δ
- source and destination cluster labels
- source and destination PC1 vectors
- cluster semantic quality Q
- raw SBERT and manifold-residual document embeddings, when created by the current pipeline

Optional companion artifacts:

- `segments_by_doc.pkl` enables text evidence and selected 3D point-cloud re-embedding.
- `morphism_comparison.pkl` enables match-level evidence and enriched diagnostics in other tabs.
- `morphism_arrangement_experiment.pkl` can be loaded to restore a previously built arrangement experiment.

## Arrangement build sequence

The experiment performs the following steps:

```text
document_delta_dict.pkl
→ extract all directed non-self cluster edges
→ compute one shape-feature vector per edge
→ cluster edge features into shape bins
→ assign every edge to a shape bin
→ accumulate document × shape-bin membership
→ compare document profiles to collection labels
```

Each directed edge is represented by:

```text
v = unit(Δ)              displacement direction
s = unit(PC1_src)        source-cluster principal direction
t = unit(PC1_dst)        destination-cluster principal direction
```

The core shape features are:

```text
cos(v, s)                source-flow alignment
cos(-v, t)               destination/incoming-flow alignment
cos(s, t)                endpoint PC1 concordance
log(||Δ||)               optional displacement length feature
```

When `include_v_bin=false`, the shape bin is primarily an edge-local form category. Edges in different residual-space locations can be grouped together if their internal source-flow-destination geometry is similar. When `include_v_bin=true`, a coordinate-dependent coarse displacement-direction bin is also appended.

## Build controls

### `shape_k`

Number of learned morphism-shape categories.

- Lower values produce broader categories.
- Higher values produce narrower categories but may fragment weak patterns.
- Typical exploratory range: 32–64.

### `include_length`

Adds displacement length to the shape feature vector.

- `true`: shape bins distinguish short vs long residual transitions.
- `false`: bins focus on angular source-flow-destination form.

Current experiments have found strong classification behavior with `include_length=true`.

### `include_v_bin`

Adds a coarse direction-bin feature for the displacement direction.

- `false`: bins emphasize rotation-stable edge-local form.
- `true`: bins combine edge-local form with coarse absolute residual-space direction.

With default v-bin settings, turning this on adds 72 one-hot dimensions from `vbin_az=12` and `vbin_el=6`. This changes `shape_feature_dim` from 4 to 76 when `include_length=true`.

### `dir_weight_beta`

Controls how strongly an edge’s membership weight depends on source/destination PC1 coherence with the displacement flow.

- `0.0`: membership reflects edge occurrence/weight without rewarding PC1-flow coherence.
- `0.5`: mild directional weighting.
- `1.0`: balanced flow-coherence weighting.
- `2.0+`: strong selectivity for coherent edge-flow geometry.

Recent arrangement experiments have shown useful collection-type classification with `dir_weight_beta=0.0`, `include_length=true`, and `include_v_bin=false`. In that setting, PC1-flow relationships are measured as shape features, but are not used to multiply edge membership weight.

### `max_edges_per_doc`

Caps the number of directed morphism edges retained per document. For typical k=5–8 CDMs this is usually not binding, because each document contributes `k × (k − 1)` directed edges.

### `min_weight`

Filters out edge records below a computed edge weight. This acts before the document membership matrix is built.

### `weight_mode`

Controls how an edge contributes to its assigned bin.

- `weighted`: add the edge’s computed weight.
- `count`: add 1 per edge.

### `normalize membership`

If true, divides each document row by its row sum. This converts each document into a distribution over shape bins.

Use `true` when comparing documents by relative morphism-action composition. Use `false` only when total morphism mass or document size should influence the comparison.

### `similarity`

Metric used for document-profile similarity.

- `cosine`: robust similarity between nonnegative membership vectors.
- `jensen_shannon`: distributional similarity when membership rows are normalized.

### `doc top-k`

Number of nearest neighbors retained per document in the document graph by shared shape membership. This affects graph density, not shape-bin creation.

### `support ≥`

Membership threshold used to decide whether a shape is present in a document for support counts and co-occurrence graphs. This affects support/co-occurrence interpretation, not shape-bin creation.

### Collection labels

Collection labels can be extracted from:

- prefix before underscore
- regex capture group
- metadata CSV
- none / Unknown

Default regex:

```regex
^([^_]+_[^_]+)_
```

Example:

```text
u0002_1234567_0000002_0001 → u0002_1234567
```

A metadata CSV should include `doc_id` and `collection_type`, or the platform will use the first two columns.

## Arrangement experiment subviews

### Summary

Displays experiment parameters, document counts, collection counts, shape counts, extracted edge counts, membership matrix dimensions, and high-level summaries.

### Shape atlas / representative edges

Lists learned shape bins and representative edges.

Shape summary columns include:

- shape ID
- edge count
- document support
- support document count
- top collection type
- collection entropy
- mean edge weight
- mean edge quality
- mean Δ length
- centroid values for the core shape features

Representative edge rows include document ID, collection type, source/destination labels, edge weight, quality, Δ length, and feature values.

### Membership heatmap

Displays the document × shape-bin membership matrix.

- X axis: shape bins.
- Y axis: documents.
- Cell color: membership weight.
- Left-hand strip: collection label blocks.
- Rollover readout: x/y cell, document ID, collection, shape ID, and membership value.

Row-order options:

- `collection_dominant_shape`
- `collection_doc_id`
- `doc_id`
- `dominant_shape`
- `membership_pca`
- `collection_membership_pca`
- `random_within_collection`

The default row order is tuned for mixed-collection arrangement experiments: group by collection, then sort within each block by dominant shape and document ID.

### Similar documents

For a selected document, lists nearest neighbors by shape-membership similarity, showing:

- rank
- document ID
- collection label
- similarity
- same-collection flag
- top shared shape categories
- top differing shape categories

### Document graph

Builds a top-k graph where nodes are documents and edge weights are similarities between document shape-membership profiles. This graph expresses shared morphism-action composition, not raw SBERT document proximity.

### Shape co-occurrence graph

Builds a graph where nodes are shape bins and edges indicate that shape categories co-occur in the same documents above the support threshold.

This is useful for detecting morphism-action bundles.

### Shape neighbor inspector

For a selected shape bin, computes three neighbor families:

1. **Centroid nearest neighbors** — nearby bins in shape-feature centroid space.
2. **Co-occurrence neighbors** — bins that occur in the same documents.
3. **Collection-profile neighbors** — bins with similar distribution across collection types.

This is useful when top-shape lists show substitutions among related collections. A substituted shape may be a centroid neighbor, a co-occurrence neighbor, or a collection-profile analogue.

## Shape Bin Field workspace

The Shape Bin Field workspace treats one selected shape bin as an inspectable morphism field.

### Purpose

It answers:

```text
What kind of morphism-action form does this bin contain?
Is the bin coherent or diffuse?
Is it generic or collection-specific?
Does it group structurally similar morphisms across semantically diverse documents?
Does it contain collection-specific subvariants?
```

### Controls

- selected shape bin
- collection/member filter
- color by collection, edge quality, prototype distance, Δ length, source-flow alignment, or destination-flow alignment
- length mode: preserve length or normalize length
- residual mode: centroid positions or centroid directions
- maximum canonical members
- maximum residual context examples
- PC1 scale
- minimum Q
- prototype overlay
- PC1 cone summary

### Scientific report

The report panel summarizes five NSF-relevant features:

1. **Generic vs collection-specific shapes** — collection entropy, collection specificity, collection lift/support.
2. **Tight vs loose shape families** — within-bin dispersion, prototype distance, PC1 cone spread.
3. **Semantically broad but structurally stable bins** — structural coherence across diverse documents/collections and document-cosine baselines where available.
4. **Recurring asymmetries in transition structure** — source-flow vs destination-flow balance, endpoint concordance, asymmetry index.
5. **Bin stratification by collection family** — collection-specific variants within a broad morphism family.

### Canonicalized morphism field

Aligns all member edges into a common edge-local coordinate frame:

- displacement points along the canonical flow axis
- source endpoint is placed on the negative side
- destination endpoint is placed on the positive side
- source PC1 is oriented with outgoing flow
- destination PC1 is oriented against incoming flow
- optional prototype/medoid overlay
- optional PC1 cones

This is the defining-form view for bins built with `include_v_bin=false`.

### Residual-space morphism context

Uses `document_delta_dict.pkl` to show selected representative bin members in their original document-centered CDM residual context.

Modes:

- `centroid_positions`: plot true residual centroid positions.
- `centroid_directions`: project centroid directions for sphere-like orientation comparison.

This is the context view: it shows where a shape occurs in its original document geometry.

### Feature distributions

Plots distributions for:

- `cos(v, source PC1)`
- `cos(-v, destination PC1)`
- `cos(source PC1, destination PC1)`
- Δ length
- edge quality
- prototype distance

### Collection composition

Shows collection contributions to the selected shape bin as counts, weighted mass, support, and enrichment-style summaries.

### Member / metrics tables

The member table lists bin edges with:

- shape row
- document ID
- collection type
- source/destination labels
- weight
- edge quality
- Δ length
- source-flow alignment
- destination-flow alignment
- endpoint concordance
- prototype distance

The metrics table reports numeric and interpretive summaries.

## Collection summary

The collection summary panels support mixed-collection experiments.

### Dominant-shape distribution

Computes the dominant shape for each document and summarizes dominant shape fractions by collection type.

### Collection-profile similarity

Computes collection average shape-membership profiles and compares them with cosine and Jensen-Shannon similarity matrices.

### Collection shape profile exports

Exports collection × shape profile matrices, pairwise collection similarities, and dominant-shape distribution tables.

## Collection ROC and representation baselines

The Collection ROC tab evaluates whether document representations recover collection labels.

### One-vs-rest ROC

For each collection type:

```text
positive = documents from that collection
negative = all other documents
score = similarity to the collection profile
```

Positive documents are scored against a leave-one-out collection profile to avoid including the document in its own target profile.

### Pairwise same-vs-cross ROC

For each document pair:

```text
positive = same collection type
negative = different collection type
score = document-pair similarity under a representation
```

### Representation domains

The tab compares, when available:

- raw SBERT document cosine
- manifold-residual document cosine
- shape-membership cosine
- shape-membership Jensen-Shannon similarity

### Similarity correlations

The similarity correlation panel compares document-pair similarity score vectors across representation domains using Pearson and Spearman correlations.

This helps assess whether shape-membership arrangement is redundant with, or complementary to, raw/residual document embedding similarity.

## Saved experiment artifact

Use **Save experiment PKL** to write:

```text
morphism_arrangement_experiment.pkl
```

This artifact stores:

- experiment kind/version
- parameters
- document IDs
- collection labels
- shape records
- shape features
- shape labels
- shape centroids
- document × shape membership matrix
- shape summaries
- representative edges
- document similarity graph data
- shape co-occurrence graph data
- shape neighbor tables
- collection summaries
- dominant-shape distribution
- collection-profile similarity
- collection ROC and representation baseline rows
- shape-bin field summaries

## Exported tables

The **Export tables** button writes reviewable CSV files, including:

- `shape_summary.csv`
- `representative_edges.csv`
- `collection_summary.csv`
- `doc_shape_membership.csv`
- `dominant_shape_distribution.csv`
- `collection_profile_similarity_pairs.csv`
- `collection_shape_profile_matrix.csv`
- `shape_centroid_neighbors.csv`
- `shape_cooccurrence_neighbors.csv`
- `shape_collection_profile_neighbors.csv`
- `shape_bin_field_summary.csv`
- `collection_roc_all_rows.csv`
- `collection_roc_one_vs_rest.csv`
- `collection_roc_pairwise.csv`
- `collection_roc_similarity_correlations.csv`

## Recommended first-phase settings

For current mixed-collection OCR experiments, a strong baseline configuration has been:

```text
shape_k = 48
include_length = true
include_v_bin = false
dir_weight_beta = 0.0
max_edges_per_doc = 2000
min_weight = 0.0
weight_mode = weighted
normalize membership = true
similarity = cosine
doc_top_k = 5
support ≥ 0.03
collection label source = doc_id_regex
collection regex = ^([^_]+_[^_]+)_
```

Interpretation of this setting:

- shape bins represent edge-local transition geometry plus displacement scale;
- absolute residual-space direction is not part of the bin definition;
- PC1-flow coherence is measured but not used to weight document membership;
- normalized membership compares documents by relative morphism-action composition.

## Interpretation cautions

- Shape IDs such as `S3` and `S12` are arbitrary learned cluster labels; numeric order does not imply proximity.
- Batch-local shape bins are learned from the current experiment. Single-collection runs will subdivide that collection’s own morphism-edge distribution rather than preserve empty bins from a mixed reference space.
- For cross-run absence/presence claims, develop or load a fixed empirical shape atlas.
- `include_v_bin=true` increases the feature dimension and introduces coordinate-dependent direction categories.
- ROC/AUC curves evaluate ranking scores, not calibrated probabilities.
- Shape-membership profiles are derived from SBERT-based CDM geometry. They should be interpreted as structural abstractions rather than independent alternatives to embeddings.
- PCA and graph layouts are visualization aids. Numerical match and arrangement values should be read from the tables and exported metrics.
