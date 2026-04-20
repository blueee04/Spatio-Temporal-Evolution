# EPIC Database Quick Reference

## One-Liner File Structure

```
Raw EPIC CSV (260 files)
    ↓ preprocess_dataset.py
Processed NPZ (260 per-embryo tensors)
    ↓ np.load()
X[N, d=5, T], alive_mask[N, T], edge_src/dst/t, idx_to_cell[N]
```

---

## Dimensions at a Glance

| Variable | Shape | Dtype | Meaning |
|----------|-------|-------|---------|
| **X** | (N, 5, T) | float32 | Node features: x, y, z, size, blot |
| **alive_mask** | (N, T) | bool | Cell is alive at time t |
| **edge_src** | (E,) | int32 | Edge source node |
| **edge_dst** | (E,) | int32 | Edge destination node |
| **edge_t** | (E,) | int32 | Edge timepoint |
| **idx_to_cell** | (N,) | object | Cell name for node index |

### Example (CD011605_5a_bright):
- N = 688 cells
- d = 5 features
- T = 210 timepoints
- E ≈ 56,605 edges

---

## Minimal Working Example

```python
import numpy as np

# Load
npz = np.load("dataset/processed/by_embryo/CD011605_5a_bright.npz", allow_pickle=True)

# Access tensors
X = npz["X"]                    # (688, 5, 210)
alive_mask = npz["alive_mask"]  # (688, 210)
edges_src = npz["edge_src"]     # (~56k,)
edges_dst = npz["edge_dst"]
edges_t = npz["edge_t"]
idx_to_cell = npz["idx_to_cell"] # (688,) with cell names

# Get features at time t=100
t = 100
X_t = X[:, :, t]  # (688, 5) — all features, all cells at t=100
alive_t = alive_mask[:, t]  # (688,) — which cells are born

# Extract only alive cells
alive_idx = np.where(alive_t)[0]
X_active = X[alive_idx, :, t]  # (M, 5) — only M live cells

print(f"Time {t}: {len(alive_idx)} cells alive")
print(f"X features (mean): {X_active.mean(axis=0)}")
```

---

## Feature Definitions

| Feature | Dimension | Unit | Range | Biological Meaning |
|---------|-----------|------|-------|---------------------|
| **x** | X-axis | pixels | 0–512 | Horizontal position |
| **y** | Y-axis | pixels | 0–512 | Vertical position |
| **z** | Z-axis (depth) | μm | 0–200 | Distance from focal plane |
| **size** | Volume | AU | 10–5000 | Cell volume / morphology |
| **blot** | Fluorescence | AU | 100–10M | Cell identity marker |

---

## Masked Cells (Growing Embryo)

**Unborn cells are all zeros:**

```python
# Cell not yet born
X[unborn_idx, :, t] == [0, 0, 0, 0, 0]  # True

# Use alive_mask to filter
if alive_mask[cell_idx, t]:
    # Safe to use X[cell_idx, :, t]
    features = X[cell_idx, :, t]
```

---

## Cell Name ↔ Index Mapping

```python
# Get index from name
cell_to_idx = {cell: idx for idx, cell in enumerate(idx_to_cell)}
ABa_idx = cell_to_idx["ABa"]  # → 1

# Get name from index
cell_name = idx_to_cell[ABa_idx]  # → "ABa"

# Lineage parsing
for cell_name in idx_to_cell:
    if len(cell_name) > 1 and cell_name[-1].isalpha():
        parent = cell_name[:-1]  # Remove last char
        # e.g., "ABal" → parent is "ABa"
```

---

## Edge List → Adjacency Matrix

```python
# Dense adjacency at time t
N = X.shape[0]
A_t = np.zeros((N, N))
mask = edge_t == t
A_t[edge_src[mask], edge_dst[mask]] = 1

# Sparse CSR (memory efficient)
from scipy.sparse import csr_matrix
A_t_sparse = csr_matrix((
    np.ones(mask.sum()),
    (edge_src[mask], edge_dst[mask])
), shape=(N, N))
```

---

## Contact Graph vs. Lineage Graph

```python
# Separate edges by type
# Spatial (undirected): distance < 20 μm
# Lineage (directed): parent→daughter

def split_edges(edge_src, edge_dst, edge_t, idx_to_cell):
    spatial_edges, lineage_edges = [], []
    
    for s, d in zip(edge_src, edge_dst):
        src_name = idx_to_cell[s]
        dst_name = idx_to_cell[d]
        
        # Lineage: daughter = parent + one letter
        if (len(dst_name) == len(src_name) + 1 and 
            src_name == dst_name[:-1]):
            lineage_edges.append((s, d))
        else:
            spatial_edges.append((s, d))
    
    return spatial_edges, lineage_edges

spatial, lineage = split_edges(edge_src, edge_dst, edge_t, idx_to_cell)
print(f"Spatial edges: {len(spatial)}")     # ~45k
print(f"Lineage edges: {len(lineage)}")    # ~5k
```

---

## Time-Windowed Analysis

```python
# Extract features from time window [t_start, t_end)
t_start, t_end = 50, 100
X_window = X[:, :, t_start:t_end]         # (N, 5, 50)
alive_window = alive_mask[:, t_start:t_end]  # (N, 50)

# Edges in window
mask = (edge_t >= t_start) & (edge_t < t_end)
edges_in_window = (edge_src[mask], edge_dst[mask], edge_t[mask])

print(f"Edges in window [{t_start}, {t_end}): {mask.sum()}")
```

---

## Batch Loading All Embryos

```python
from pathlib import Path

embryos = []
for npz_path in sorted(Path("dataset/processed/by_embryo").glob("*.npz")):
    npz = np.load(npz_path, allow_pickle=True)
    embryos.append({
        "name": npz_path.stem,
        "X": npz["X"],
        "alive": npz["alive_mask"],
        "edge_src": npz["edge_src"],
        "edge_dst": npz["edge_dst"],
        "edge_t": npz["edge_t"],
        "idx_to_cell": npz["idx_to_cell"],
    })

print(f"Loaded {len(embryos)} embryos")
```

---

## Replicability & Reproducibility

```python
# Every NPZ contains source info
source_file = str(npz["source_file"])        # "CD011605_5a_bright.csv"
t0 = int(npz["t0"])                          # Usually 1
T = int(npz["T"])                            # 210 for example
absolute_time = t0 + t_idx                   # Convert 0-indexed → 1-indexed
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'X'` | Wrong .npz file | Use `*.npz` from `by_embryo/` |
| Shape mismatch | Comparing embryos | Check N varies per embryo; use per-file metadata |
| Out of memory | Loading all 260 simultaneously | Load per-embryo or sample subset |
| Zero features but `alive_mask==True` | Data quality | Usually means the CSV row was corrupted |
| Negative coordinates | Measurement error | Clamp to 0 or filter with `X > 0` |

---

## Debugging Checklist

```python
npz = np.load("file.npz", allow_pickle=True)

# ✓ Shapes match
assert npz["X"].shape[:2] == (npz["X"].shape[0], 5)
assert npz["alive_mask"].shape[0] == npz["X"].shape[0]

# ✓ Edges valid
assert npz["edge_src"].max() < npz["X"].shape[0]
assert npz["edge_dst"].max() < npz["X"].shape[0]
assert npz["edge_t"].max() < npz["X"].shape[2]

# ✓ Unborn cells zero
unborn = ~npz["alive_mask"]
assert (npz["X"][unborn] == 0).all()

# ✓ Cell name count
assert len(npz["idx_to_cell"]) == npz["X"].shape[0]
assert len(set(npz["idx_to_cell"])) == len(npz["idx_to_cell"])

print("✓ All checks passed")
```

---

## Performance Tips

- **Don't load all 260 files:** Load per-embryo or batch (10–50 at a time)
- **Use sparse graphs:** Keep `(edge_src, edge_dst, edge_t)` sparse; build dense A[t] on-demand
- **Mask early:** Filter with `alive_mask` before analysis to avoid zero-padding noise
- **Chunk time:** Process in time windows (e.g., 50 timepoint chunks)
- **Cache idx_to_cell:** Build once, reuse across analyses

---

## Metadata Verification

```bash
# Count processed embryos
ls -1 dataset/processed/by_embryo/*.npz | wc -l
# Expected: 260

# Check manifest
wc -l dataset/processed/by_embryo/manifest.txt
# Expected: 260 (one file per line)
```

---

## Citation

If using this dataset in research, cite the original EPIC Consortium and Sulston et al.

```bibtex
@article{sulston1983lineage,
  title={The embryonic cell lineage of the nematode {C}aenorhabditis elegans},
  author={Sulston, JE and Schierenberg, E and White, JG and Thomson, JN},
  journal={Developmental Biology},
  volume={100},
  number={1},
  pages={64--119},
  year={1983}
}
```
