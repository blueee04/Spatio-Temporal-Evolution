# EPIC Database Schema Reference

## Dataset Specification

**Project:** Spatio-Temporal Evolution (C. elegans EPIC embryos)  
**Format Version:** 1.0  
**Schema Date:** April 2026  

---

## Overview

```
Raw Data (260 CSV files)
         ↓
Per-embryo processing
         ↓
Processed Data (260 NPZ archives)
```

---

## 1. Input Schema: Raw EPIC CSV

### File Format
- **Location:** `dataset/raw/*.csv`
- **Format:** CSV (comma-delimited)
- **Count:** 260 files (one per embryo)
- **Encoding:** UTF-8

### Column Specification

| Column | Type | Unit | Range | Required | Used? | Notes |
|--------|------|------|-------|----------|-------|-------|
| `cellTime` | string | — | — | ✓ | ✗ | Row ID (e.g., "AB:1"); duplicate of cell:time |
| `cell` | string | — | — | ✓ | ✓ | C. elegans cell name (e.g., "AB", "ABal", "Zrp1aaa") |
| `time` | integer | frame | 1–210 | ✓ | ✓ | Absolute timepoint (1-indexed) |
| `none` | integer | — | — | ✓ | ✗ | Unknown field; ignored |
| `global` | integer | — | — | ✓ | ✗ | Global offset; unused |
| `local` | integer | — | — | ✓ | ✗ | Local offset; unused |
| `blot` | float | AU | 100–10M | ✓ | ✓ | **Primary identifier:** fluorescence intensity |
| `cross` | float | — | — | ✓ | ✗ | Cross-correlation; unused |
| `z` | float | μm | 0–200 | ✓ | ✓ | Depth (focal plane offset) |
| `x` | float | pixel | 0–512 | ✓ | ✓ | Horizontal position (X-axis) |
| `y` | float | pixel | 0–512 | ✓ | ✓ | Vertical position (Y-axis) |
| `size` | float | AU | 10–5000 | ✓ | ✓ | Cell volume / morphological size |
| `gweight` | float | — | — | ✓ | ✗ | Image intensity weight; unused |

### Constraints
- **Unique key:** (cell, time) — only one measurement per cell per timepoint
- **Cell naming:** C. elegans nomenclature compliant (alphanumeric + letters encode lineage)
- **Time ordering:** Rows typically in chronological order but not guaranteed
- **No missing values:** All columns present in every row (though some unused)
- **Sparse data:** Only living cells appear; births create new rows

### Example Record
```csv
cellTime,cell,time,none,global,local,blot,cross,z,x,y,size,gweight
ABal:25,ABal,25,22722,-2278,10,815432,-2278,19.2,166,257,74,815432
```

---

## 2. Processing Schema: Transformation Rules

### Input Transformation

#### Cell Name Normalization
```
Input:  arbitrary string from CSV
ProcessProcessing:
  1. Strip whitespace
  2. Ensure alphabetic characters only (or underscores for legacy names)
  3. Preserve case (C. elegans uses ABa vs ABp, not aba vs abp)
Output: cell_to_idx mapping (sorted alphabetically)
```

#### Time Indexing
```
Input:  time (1-indexed, 1 to T_max)
Processing:
  1. t0 = min(time_in_csv)
  2. T = max(time_in_csv) - t0 + 1
  3. t_idx = time - t0  (convert to 0-indexed)
Output: t_idx in range [0, T-1]
```

#### Feature Extraction
```
Input:  x, y, z, size, blot columns
Processing:
  - Convert all to float32
  - Verify non-negative (except z can be ~0)
  - Stack into features array [x, y, z, size, blot]
Output:  X[c_idx, :, t_idx] = [x, y, z, size, blot]
```

#### Alive Masking
```
Input:  presence of (cell, time) row in CSV
Processing:
  - If row exists: alive_mask[c_idx, t_idx] = True
  - If no row:      alive_mask[c_idx, t_idx] = False (never populated)
Output:  alive_mask[N, T] boolean array
```

#### Edge Construction

**Spatial edges (undirected proximity):**
```
Input:  (x, y, z) coordinates of all alive cells at time t
Processing:
  1. Compute pairwise Euclidean distance
  2. For each pair (i, j) with distance < threshold (20 μm):
     - Add edge i → j
     - Add edge j → i (undirected)
  3. Store as (edge_src, edge_dst, edge_t) triplets
Output: ~45,000 edges per embryo (mostly spatial)
```

**Lineage edges (directed ancestry):**
```
Input:  cell names at time t
Processing:
  1. For each cell alive at time t:
     - If last character is alphabetic:
        parent = cell[:-1]  (remove last character)
        if parent exists and is alive:
           add edge: parent → cell
  2. Store as (edge_src, edge_dst, edge_t) triplets
Output: ~5,000 edges per embryo (genealogy)
Caveat: Only applies when last char is alpha (not for "P1", cells with underscores, etc.)
```

---

## 3. Output Schema: Processed NPZ Archive

### Archive Format
- **Type:** NumPy `.npz` (compressed NumPy archive, ZIP format)
- **Compression:** DEFLATE (automatic via `np.savez_compressed()`)
- **Compression ratio:** ~3–5× typical for this data

### Archive Contents

#### X: Node Feature Tensor

**Path in NPZ:** `X`  
**Type:** `np.ndarray`  
**DType:** `float32`  
**Shape:** `(N, d, T)` where:
- N = number of cells (varies per embryo, typically 600–750)
- d = 5 (fixed feature dimension)
- T = number of timepoints (varies per embryo, typically 150–250)

**Semantics:**
```
X[c_idx, f_idx, t_idx] = value

where:
  c_idx ∈ [0, N-1]           — cell index
  f_idx ∈ {0,1,2,3,4}        — feature: 0=x, 1=y, 2=z, 3=size, 4=blot
  t_idx ∈ [0, T-1]           — timepoint (0-indexed)
  value ∈ ℝ or 0             — feature value or 0 if unborn
```

**Constraints:**
```
if alive_mask[c_idx, t_idx] == False:
    X[c_idx, :, t_idx] == [0, 0, 0, 0, 0]  (all zeros)
    
if alive_mask[c_idx, t_idx] == True:
    X[c_idx, 0, t_idx] ∈ [0, ~512]          (x, pixels)
    X[c_idx, 1, t_idx] ∈ [0, ~512]          (y, pixels)
    X[c_idx, 2, t_idx] ∈ [0, ~200]          (z, micrometers)
    X[c_idx, 3, t_idx] ∈ [10, ~5000]        (size, arbitrary units)
    X[c_idx, 4, t_idx] ∈ [100, 10M]         (blot, arbitrary units)
```

**Feature Definitions:**
| Index | Name | Unit | Physical Meaning |
|-------|------|------|------------------|
| 0 | x | pixels | Horizontal position |
| 1 | y | pixels | Vertical position |
| 2 | z | micrometers | Depth from focal plane |
| 3 | size | AU | Cell volume / morphology |
| 4 | blot | AU | Fluorescence intensity (identity marker) |

---

#### alive_mask: Birth/Alive Tracking

**Path in NPZ:** `alive_mask`  
**Type:** `np.ndarray`  
**DType:** `bool`  
**Shape:** `(N, T)`

**Semantics:**
```
alive_mask[c_idx, t_idx] = True   if cell is observed at this timepoint
alive_mask[c_idx, t_idx] = False  if cell not yet born or unobserved
```

**Constraints:**
```
Monotonicity: Once False, always False before first True (no resurrection)
if alive_mask[c_idx, t] == True and t' < t:
    then ∃ t0 such that:
        alive_mask[c_idx, 0:t0] == False
        alive_mask[c_idx, t0:] == True
```

**Use Cases:**
1. Identify cell birth time: `t_birth = argmax(alive_mask[c_idx])`
2. Extract valid features: `X_valid = X[alive_mask]`
3. Compute lifespan: `lifespan = alive_mask.sum(axis=1)`

---

#### edge_src, edge_dst, edge_t: Sparse Edge Lists

**Paths in NPZ:**
- `edge_src` — source node indices
- `edge_dst` — destination node indices
- `edge_t` — timepoint indices

**Type:** `np.ndarray`  
**DType:** `int32`  
**Shape:** `(E,)` where E ≈ 56,000 per embryo

**Semantics:**
```
An edge exists between node edge_src[k] → edge_dst[k] at time edge_t[k]
if and only if k ∈ [0, E-1]
```

**Constraints:**
```
edge_src[k] ∈ [0, N-1]    for all k
edge_dst[k] ∈ [0, N-1]    for all k
edge_t[k]   ∈ [0, T-1]    for all k

Assumption: No duplicate edges (same src, dst, t)
```

**Edge Type Classification:**

To distinguish spatial from lineage edges:
```python
def classify_edge(src_idx, dst_idx, idx_to_cell):
    src_name = idx_to_cell[src_idx]
    dst_name = idx_to_cell[dst_idx]
    
    # Lineage: daughter is parent + one letter
    if (len(dst_name) == len(src_name) + 1 and 
        dst_name.startswith(src_name)):
        return "lineage"
    else:
        return "spatial"
```

---

#### idx_to_cell: Cell Name Mapping

**Path in NPZ:** `idx_to_cell`  
**Type:** `np.ndarray`  
**DType:** `object` (Python strings)  
**Shape:** `(N,)`

**Semantics:**
```
idx_to_cell[c_idx] = cell_name (string)

Example:
  idx_to_cell[0]   = "AB"
  idx_to_cell[1]   = "ABa"
  idx_to_cell[2]   = "ABal"
  ...
  idx_to_cell[687] = "Zrp1aaa"
```

**Constraints:**
```
All unique (no duplicate cell names)
All alphabetic characters (and hyphens for legacy names)
Sorted alphabetically (for reproducibility)
len(idx_to_cell) == N
```

**Inverse Mapping (derived):**
```python
cell_to_idx = {cell: idx for idx, cell in enumerate(idx_to_cell)}
```

---

#### t0, T: Time Metadata

**Paths in NPZ:**
- `t0` — first absolute timepoint
- `T` — total number of timepoints

**Type:** `np.ndarray` (scalar or 0-d array)  
**DType:** `int32`  
**Shape:** `()` (scalar)

**Semantics:**
```
t0 = t_min from CSV (typically 1)
T  = t_max - t_min + 1

Conversion:
  t_absolute = t0 + t_index
  t_index = t_absolute - t0
```

**Example:**
```
Raw CSV times: 1, 2, 3, ..., 210
t0 = 1
T  = 210
t_index = 24 → t_absolute = 1 + 24 = 25
```

---

#### source_file: Provenance

**Path in NPZ:** `source_file`  
**Type:** `np.ndarray`  
**DType:** `object` (Python string)  
**Shape:** `()` (scalar)

**Value:** Original CSV filename (e.g., "CD011605_5a_bright.csv")

**Use:** Trace back to raw data for reprocessing or verification

---

### NPZ Summary Table

| Field | Shape | DType | Size (MB) | Purpose |
|-------|-------|-------|-----------|---------|
| X | (N, 5, T) | float32 | ~13 | Node features |
| alive_mask | (N, T) | bool | ~0.2 | Birth tracking |
| edge_src | (E,) | int32 | ~0.2 | Edge list |
| edge_dst | (E,) | int32 | ~0.2 | Edge list |
| edge_t | (E,) | int32 | ~0.2 | Edge list |
| idx_to_cell | (N,) | object | ~0.02 | Name mapping |
| t0 | () | int32 | ~0.00001 | Metadata |
| T | () | int32 | ~0.00001 | Metadata |
| source_file | () | object | ~0.00003 | Provenance |
| **Total (uncompressed)** | — | — | **~14** | — |
| **Total (compressed)** | — | — | **~0.7** | ~20× compression |

---

## 4. Relationships & Constraints

### Dimension Consistency
```
X.shape[0] == alive_mask.shape[0] == len(idx_to_cell) == N
X.shape[1] == 5  (features)
X.shape[2] == alive_mask.shape[1] == T
```

### Index Validity
```
∀ k ∈ [0, E-1]:
  edge_src[k] < N
  edge_dst[k] < N
  edge_t[k] < T
```

### Data Integrity
```
∀ c ∈ [0, N-1], t ∈ [0, T-1]:
  if alive_mask[c, t] == False:
    X[c, :, t] == [0, 0, 0, 0, 0]
```

### Uniqueness
```
len(unique(idx_to_cell)) == N  (no duplicate names)
∀ (src, dst, t): unique triplets (no multi-edges)
```

### Temporal Ordering
```
t0 ≤ t0 + 1 ≤ ... ≤ t0 + T - 1  (times monotonic)
```

### Lineage Consistency
```
∀ lineage edge (parent_idx, daughter_idx):
  parent_name = idx_to_cell[parent_idx]
  daughter_name = idx_to_cell[daughter_idx]
  daughter_name == parent_name + single_letter
```

---

## 5. Serialization Details

### NPZ Format (ZIP-based)
```
file.npz (actually a ZIP archive)
├── X.npy
├── alive_mask.npy
├── edge_src.npy
├── edge_dst.npy
├── edge_t.npy
├── idx_to_cell.npy
├── t0.npy
├── T.npy
└── source_file.npy
```

### Loading
```python
import numpy as np

# Load
npz = np.load("file.npz", allow_pickle=True)

# Access (two equivalent ways)
X = npz["X"]  # Dictionary-like
X = npz.files["X"]  # Via file listing

# Close (optional but recommended)
npz.close()
```

### Saving
```python
import numpy as np

np.savez_compressed(
    "file.npz",
    X=X,
    alive_mask=alive_mask,
    edge_src=edge_src,
    edge_dst=edge_dst,
    edge_t=edge_t,
    idx_to_cell=idx_to_cell,
    t0=np.int32(t0),
    T=np.int32(T),
    source_file=source_file
)
```

---

## 6. Data Quality Standards

### Validation Checklist

**Shape consistency** (must pass)
```python
assert X.shape[0] == alive.shape[0] == len(idx_to_cell)
assert X.shape[1] == 5
assert X.shape[2] == alive.shape[1]
assert len(edge_src) == len(edge_dst) == len(edge_t)
```

**Index validity** (must pass)
```python
assert 0 <= edge_src.min() and edge_src.max() < N
assert 0 <= edge_dst.min() and edge_dst.max() < N
assert 0 <= edge_t.min() and edge_t.max() < T
```

**Masking consistency** (must pass)
```python
assert (X[~alive] == 0).all()  # Unborn cells all zero
```

**Name uniqueness** (must pass)
```python
assert len(np.unique(idx_to_cell)) == N
```

**Reasonable ranges** (should pass)
```python
X_alive = X[alive]
assert 0 <= X_alive[:, :3].max() < 1000  # Coordinates
assert 0 < X_alive[:, 3:].min()  # Size, blot positive
```

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 2026 | Initial schema definition; per-embryo tensors, sparse edges |

---

## 8. References

**Raw EPIC data source:** http://epic.gs.washington.edu/  
**C. elegans nomenclature:** Sulston et al. (1983)  
**Distance threshold:** 20 micrometers (standard contact distance)

---

**Schema Document Version:** 1.0  
**Last Updated:** April 2026
