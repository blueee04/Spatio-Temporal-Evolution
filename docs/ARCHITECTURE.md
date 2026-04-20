# Data Pipeline & Architecture Documentation

## System Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │   C. elegans Embryo Microscopy      │
                    │   (EPIC fluorescence video)          │
                    └──────────────┬──────────────────────┘
                                   │
                                   ↓
                    ┌─────────────────────────────────────┐
                    │    Raw EPIC CSV Files (260)         │
                    │  dataset/raw/*.csv                   │
                    │                                      │
                    │  Columns:                            │
                    │  cell, time, x, y, z, size, blot    │
                    │  (+ unused metadata columns)         │
                    │                                      │
                    │  Format: N×T sparse table            │
                    │  (only alive cells per timepoint)    │
                    └──────────────┬──────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                              │
    [preprocess_dataset.py]                              │
    python scripts/                                      │
      preprocess_dataset.py                              │
      --raw_dir dataset/raw                              │
      --out dataset/processed/by_embryo                  │
      --distance_threshold 20                            │
            │                                             │
            └──────────────────────┬───────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │  Per-Embryo Processing            │
                    │                                   │
                    │  src/epic_preprocess.py:          │
                    │  • build_local_index()            │
                    │  • populate_X()                   │
                    │  • populate_alive_mask()          │
                    │  • build_spatial_edges()          │
                    │  • build_lineage_edges()          │
                    │                                   │
                    │  Input:  Raw CSV (1 embryo)       │
                    │  Output: X, A_sparse, metadata    │
                    └──────────────┬────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ↓                          ↓                          ↓
  [Spatial Edges]          [Lineage Edges]            [Features & Mask]
  Proximity graph          Parent→daughter           Node features X
  (undirected)             (directed)                 & birth tracking
  distance < 20 μm         naming convention         alive_mask
  ~45k edges/embryo        ~5k edges/embryo
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │   Compressed NPZ Archive          │
                    │   dataset/processed/by_embryo/    │
                    │   *.npz (260 files)               │
                    │                                   │
                    │  npz.keys():                      │
                    │  • X[N, 5, T]                     │
                    │  • alive_mask[N, T]               │
                    │  • edge_src, edge_dst, edge_t     │
                    │  • idx_to_cell[N]                 │
                    │  • t0, T, source_file             │
                    │                                   │
                    │  Per-file storage: ~0.7 MiB       │
                    │  (compressed sparse graph)        │
                    └──────────────┬────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ↓                          ↓                          ↓
  [Downstream Tasks]        [Analysis Queries]        [Visualization]
  • GNN training             • Cell trajectories        • Graph plots
  • Node classification      • Lineage ancestry        • Heatmaps
  • Link prediction          • Spatial statistics      • 3D tracks
  • Temporal dynamics        • Birth/division times    • Feature dist.
```

---

## Data Structures & Relationships

### Tensor Dimensions

```
Per-embryo (local):
  N = 688 cells (varies by embryo)
  d = 5 features (fixed)
  T = 210 timepoints (varies by embryo)

D:
  X[N, d, T]           — Node features (float32)
  alive_mask[N, T]     — Birth/alive tracking (bool)
  edge_src[E]          — Source node indices (int32)
  edge_dst[E]          — Destination node indices (int32)
  edge_t[E]            — Edge timepoints (int32)
    E ≈ 56,605 per embryo
  idx_to_cell[N]       — Cell name mapping (object)
```

### Mapping Between Worlds

```
Cell Naming (C. elegans standard)
  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
  AB → ABal → ABalp (divisional tree)
  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
Node Indexing (sequential 0..N-1)
  idx_to_cell mapping: 0↔"AB", 1↔"ABa", 2↔"ABal", ...

Biological Time (EPIC timestamps)
  1, 2, 3, ..., 210 (absolute seconds ≈ × 4 sec/frame)
  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
Computational Time (0-indexed)
  0, 1, 2, ..., 209 (standard array indexing)
  t_absolute = t0 + t_index
```

---

## Input CSV → Tensor Conversion

### Example: One Cell Over Time

**Raw CSV (cell "ABal"):**
```
cell    time  x     y     z      size   blot
ABal    25    166   257   19.2   74     815432
ABal    26    165   243   18.1   74     988719
ABal    27    163   239   17.1   77     1315431
...
ABal    210   162   248   18.5   72     901234
```

**Tensor representation (ABal_idx = 2):**
```
X[2, :, 24] = [166, 257, 19.2, 74, 815432]    (t_index=24 → time=25)
X[2, :, 25] = [165, 243, 18.1, 74, 988719]    (t_index=25 → time=26)
...
alive_mask[2, 24:210] = [True, True, ..., True]
alive_mask[2, 0:24]   = [False, False, ..., False]  (not yet born)
```

---

## Edge Construction Logic

### Spatial Edges (Undirected Proximity)

**Algorithm:**
```
for t in range(T):
    living_cells = [c for c if alive_mask[c, t] == True]
    coords = X[living_cells, :3, t]  # xyz coordinates
    distances = pairwise_euclidean(coords)  # (M, M)
    
    for i, cell_i in enumerate(living_cells):
        for j, cell_j in enumerate(living_cells):
            if i < j and distances[i,j] < THRESHOLD (=20):
                edge_src.append(idx[cell_i])
                edge_dst.append(idx[cell_j])
                edge_t.append(t)
                
                # Undirected: also add reverse
                edge_src.append(idx[cell_j])
                edge_dst.append(idx[cell_i])
                edge_t.append(t)
```

**Example (t=100, two cells close together):**
```
Cell "ABal" (idx=2) at (163, 239, 17.1)
Cell "ABap" (idx=3) at (168, 241, 17.3)
Distance = √((168-163)² + (241-239)² + (17.3-17.1)²) ≈ 5.4 μm < 20 μm
→ Edges: (2→3, 3→2) added with edge_t=100
```

### Lineage Edges (Directed Ancestry)

**Algorithm:**
```
for t in range(T):
    living_cells = [c for c if alive_mask[c, t] == True]
    
    for cell in living_cells:
        if last_character_is_alphabetic(cell):
            parent = cell[:-1]
            if parent in living_cells:
                edge_src.append(idx[parent])
                edge_dst.append(idx[cell])
                edge_t.append(t)
```

**Example:**
```
At any t where both "ABa" and "ABal" are alive:
  Parent: "ABa" (idx=1)
  Daughter: "ABal" (idx=2)
  Append edge: (1→2, time=t)
  
  This edge appears at every t from t_birth(ABal) onwards
  For CD011605: ABal appears at t=25, so edge (1→2) in edge_t for t≥24
```

---

## Data Flow: Alive Masking

```
┌─────────────────────────────────────────────┐
│ Birth Event: Cell "ABal" divisions at t=25 │
└────────────┬────────────────────────────────┘
             │
    ┌────────▼────────┐
    │ First appearance│
    │ in raw CSV      │
    │ at t_EPIC = 25  │
    │ (1-indexed)     │
    └─────┬──────────┘
          │
    ┌─────▼────────────────────────────┐
    │ Preprocessing:                    │
    │ t_index = time - t0               │
    │         = 25 - 1                  │
    │         = 24 (0-indexed)          │
    └──────┬──────────────────────────┘
           │
    ┌──────▼────────────────────────────────┐
    │ Set alive_mask[ABal_idx, 24] = True   │
    │ Set X[ABal_idx, :, 24] = [166, ...]   │
    └──────┬────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────┐
    │ All indices [0..23]:                  │
    │   alive_mask[ABal_idx, 0..23] = False │
    │   X[ABal_idx, :, 0..23] = 0 (masked)  │
    └───────────────────────────────────────┘
```

---

## Output NPZ Structure

```
CD011605_5a_bright.npz (compressed NumPy archive)
│
├─ X: ndarray[688, 5, 210] dtype=float32
│   ├─ 688 cells (node count N)
│   ├─ 5 features per cell: [x, y, z, size, blot]
│   └─ 210 timepoints (T)
│
├─ alive_mask: ndarray[688, 210] dtype=bool
│   ├─ True if cell is observed at this timepoint
│   └─ False if not yet born or unobserved
│
├─ edge_src: ndarray[56605] dtype=int32
│   └─ Node indices (0..687) for edge sources
│
├─ edge_dst: ndarray[56605] dtype=int32
│   └─ Node indices (0..687) for edge destinations
│
├─ edge_t: ndarray[56605] dtype=int32
│   └─ Timepoint indices (0..209) for each edge
│
├─ idx_to_cell: ndarray[688] dtype=object
│   ├─ String array: ["AB", "ABa", "ABal", ..., "Zrp1aaa"]
│   └─ Reverse maps: idx → cell_name
│
├─ t0: ndarray[] dtype=int32
│   └─ First absolute timepoint (usually 1)
│
├─ T: ndarray[] dtype=int32
│   └─ Total timepoints (210)
│
└─ source_file: ndarray[] dtype=object
    └─ Original raw filename for provenance
```

---

## Spatio-Temporal Compression

### Dense Representation (Naive)
```
A[N, N, T] adjacency tensor
= 688 × 688 × 210 cells
= 99,681,120 real values
× 8 bytes per float64
= 797 MB per embryo
× 260 embryos
= 207 GB (single dense matrix!)
```

### Sparse Representation (Ours)
```
(edge_src, edge_dst, edge_t)
= 57,000 edges per embryo × 3 arrays × 4 bytes
= 0.68 MB per embryo (uncompressed)
× npz compression (~3x reduction)
= 0.23 MB per embryo (compressed)
× 260 embryos
= 60 MB total!

Reduction factor: 3,500×
```

---

## Query Pattern Examples

### Query 1: "Features of cell 'ABal' at timepoint 50"

```python
cell_name = "ABal"
t_absolute = 50

# Resolve
cell_idx = cell_to_idx[cell_name]
t_idx = t_absolute - t0  # Convert to 0-indexed

# Check if alive
if alive_mask[cell_idx, t_idx]:
    features = X[cell_idx, :, t_idx]  # [x, y, z, size, blot]
    print(f"{cell_name} at time {t_absolute}: {features}")
else:
    print(f"{cell_name} not yet born at time {t_absolute}")
```

### Query 2: "What are cell 'ABal' 's daughters?"

```python
parent = "ABal"
cell_to_idx = {c: i for i, c in enumerate(idx_to_cell)}

# Find all daughters (naming: parent + 1 letter)
daughters = [c for c in idx_to_cell if c.startswith(parent) and len(c) == len(parent)+1]
print(f"Daughters of {parent}: {daughters}")
# Output: ["ABalp", "ABalaa"]
```

### Query 3: "Which cells are touching 'ABal' at time 50?"

```python
target_idx = cell_to_idx["ABal"]
t_idx = 50 - t0

# Get all edges at this time
mask = edge_t == t_idx
edges_at_t = (edge_src[mask], edge_dst[mask])

# Find edges involving ABal
neighbors = set()
for src, dst in zip(*edges_at_t):
    if src == target_idx:
        neighbors.add(dst)
    if dst == target_idx:
        neighbors.add(src)

for neighbor_idx in neighbors:
    print(f"  {idx_to_cell[neighbor_idx]}")
```

---

## Consistency & Reproducibility

### Determinism Guarantees

1. **Cell ordering:** Alphabetically sorted → reproducible index mapping
2. **Time indexing:** Always 0-indexed internally; conversion via t0
3. **Edge deduplication:** Lineage & spatial edges stored distinctly
4. **Metadata:** Every NPZ includes t0, T, source_file for traceability

### Validation Checklist

```
✓ shapes match: X.shape[0] == alive_mask.shape[0] == len(idx_to_cell)
✓ edges valid: max(edge_src/dst) < N; max(edge_t) < T; min() ≥ 0
✓ masking consistent: X[~alive_mask] == 0 everywhere
✓ no duplicate cell names: len(unique(idx_to_cell)) == N
✓ birth monotonicity: once alive_mask[c,t]=False, stays False for earlier t
```

---

## Performance Characteristics

### Load Time (single embryo)
```
npz.load() with allow_pickle=True: ~100 ms
All tensors decompressed into RAM: ~50 MB per embryo
```

### Memory Usage
```
One embryo in RAM: ~50 MB
Batch of 10: ~500 MB
All 260 embryos: ~13 GB (uncompressed)
```

### Access Time
```
X[i, j, t] lookup: O(1) — NumPy array indexing
Edge query at time t: O(E) where E ≈ 57k — filter edge_t array
Cell name lookup: O(1) — dictionary or array index
```

### Scalability
```
Best case: Per-embryo processing (embarrassingly parallel)
Distributed setup: Load embryo-i on machine-i; aggregate results
Streaming: Process windows of time (avoid full T in memory)
```

---

**Architecture Document Version:** 1.0  
**Diagram Tool:** Plain-text Mermaid-compatible (render at mermaid.live)
