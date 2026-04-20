# EPIC Database Documentation Index

## Quick Navigation

### 📖 For First-Time Users
1. **Start here:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — 5-minute overview
2. **Then read:** [DATABASE_DOCUMENTATION.md](DATABASE_DOCUMENTATION.md) — Sections 1–3 (input, processing, output)
3. **Try this:** Run `python scripts/usage_examples.py` to see working code

### 🏗️ For System Design
- [ARCHITECTURE.md](ARCHITECTURE.md) — Data flow diagrams, tensor construction, edge building logic

### 💻 For Implementation
- [Database_DOCUMENTATION.md](#9-common-operations--recipes) — Code recipes and patterns
- [scripts/usage_examples.py](../scripts/usage_examples.py) — 7 runnable examples
- [src/epic_preprocess.py](../src/epic_preprocess.py) — Core preprocessing functions

### 🔍 For Debugging & Validation
- [DATABASE_DOCUMENTATION.md](#8-quality-control--validation) — QC checks and error handling
- [QUICK_REFERENCE.md](#debugging-checklist) — Common errors & fixes

### 📊 For Data Analysis
- [DATABASE_DOCUMENTATION.md](#7-file-manifest--provenance) — Manifest structure, batch loading
- [scripts/usage_examples.py](../scripts/usage_examples.py) — Example 6 (batch statistics)

---

## File Manifest

```
docs/
├── README.md (this file)
├── DATABASE_DOCUMENTATION.md      (Comprehensive: 11 sections, ~1000 lines)
├── QUICK_REFERENCE.md             (Quick-lookup: 10 sections, ~300 lines)
├── ARCHITECTURE.md                (Visual/technical: data flow, queries)
│
scripts/
├── preprocess_dataset.py           (Main preprocessing script)
├── usage_examples.py               (7 runnable examples)
├── make_figures.py                 (Visualization utils)
│
src/
├── epic_preprocess.py              (Core preprocessing functions)
│   ├── build_local_index()
│   ├── preprocess_epic_file_sparse()
│   └── helper functions
│
dataset/
├── raw/                            (260 *.csv files — raw EPIC input)
│   ├── CD011505_end1red_bright.csv
│   ├── CD011605_5a_bright.csv
│   ├── ... (258 more)
│
└── processed/by_embryo/            (260 *.npz files — processed output)
    ├── CD011505_end1red_bright.npz
    ├── CD011605_5a_bright.npz
    ├── ... (258 more)
    └── manifest.txt
```

---

## Document Overview

### 📄 DATABASE_DOCUMENTATION.md (Comprehensive Reference)

**Length:** ~1200 lines | **Audience:** Everyone (but especially developers)

| Section | Content | Use Case |
|---------|---------|----------|
| 1 | Raw EPIC CSV format & columns | Understanding input data |
| 2 | Preprocessing pipeline details | How data is transformed |
| 3 | Output NPZ tensors specifications | Data format reference |
| 4 | Biological interpretation | Understanding C. elegans |
| 5 | Memory & performance | System requirements |
| 6 | Usage patterns | How to load/query data |
| 7 | File manifest & provenance | Tracking & validation |
| 8 | QC & validation checks | Data quality assurance |
| 9 | Common operations & recipes | Code examples |
| 10 | Summary pipeline diagram | High-level overview |
| 11 | Troubleshooting | Error handling |

---

### 📋 QUICK_REFERENCE.md (Quick Lookup)

**Length:** ~300 lines | **Audience:** Experienced users needing fast reference

| Section | Content |
|---------|---------|
| Structure | 5-second file layout |
| Dimensions | Tensor shapes at a glance |
| MWE | Minimal working example |
| Features | Column definitions table |
| Masking | Handling unborn cells |
| Mapping | Cell name ↔ index conversions |
| Edge Lists | Sparse → dense example |
| Contact vs. Lineage | Separating edge types |
| Time Windows | Temporal slicing |
| Batch Loading | Multi-embryo patterns |
| QC Checklist | Validation tests |
| Performance | Tips & tricks |
| Metadata | Tracking info |

---

### 🏛️ ARCHITECTURE.md (System Design & Data Flow)

**Length:** ~400 lines |**Audience:** System designers, advanced users

| Section | Content |
|---------|---------|
| System Diagram | Full pipeline ASCII art |
| Tensor Relationships | Dimension mapping & constraints |
| CSV → Tensor | Example cell trajectory |
| Spatial Edge Logic | Proximity algorithm |
| Lineage Edge Logic | Ancestry naming reconstruction |
| Alive Masking | Birth tracking data flow |
| NPZ Structure | File contents breakdown |
| Sparse Compression | Memory savings analysis |
| Query Patterns | Example SQL-like queries |
| Reproducibility | Determinism guarantees |
| Performance | Benchmarks & scalability |

---

## Usage Patterns by Role

### 🧪 Data Scientist / ML Researcher

**Goal:** Train models on cell dynamics

**Read:**
1. QUICK_REFERENCE.md (Section "Minimal Working Example")
2. DATABASE_DOCUMENTATION.md (Sections 3, 6, 9)
3. scripts/usage_examples.py (Examples 1, 2, 6)

**Code pattern:**
```python
# Load batch of embryos
from pathlib import Path
import numpy as np

for npz_path in sorted(Path("dataset/processed/by_embryo").glob("*.npz"))[:10]:
    X = np.load(npz_path)["X"]  # (N, 5, T)
    # Train your model...
```

---

### 🔬 Biologist / Domain Expert

**Goal:** Understand cell behavior, verify data quality

**Read:**
1. DATABASE_DOCUMENTATION.md (Sections 1, 4, 8)
2. QUICK_REFERENCE.md (Sections "Feature Definitions", "Cell Name ↔ Index")
3. scripts/usage_examples.py (Examples 3, 5)

**Code pattern:**
```python
# Track a cell's developmental history
cell_name = "ABal"
trajectory = [X[cell_idx, :, t] for t in range(T) if alive_mask[cell_idx, t]]
# Analyze migration, division timing, etc.
```

---

### 🛠️ Software Engineer / DevOps

**Goal:** Process data, manage pipelines, ensure data quality

**Read:**
1. DATABASE_DOCUMENTATION.md (Sections 2, 5, 7, 8, 11)
2. ARCHITECTURE.md (Sections "Data Pipeline", "Validation")
3. scripts/preprocess_dataset.py (runnable script)

**Code pattern:**
```bash
# Run full preprocessing pipeline
python scripts/preprocess_dataset.py \
    --raw_dir dataset/raw \
    --out dataset/processed/by_embryo \
    --distance_threshold 20

# Verify output
python scripts/usage_examples.py  # Run all QC checks
```

---

### 📚 Student / New Contributor

**Goal:** Understand the system from scratch

**Path:**
1. **Week 1:** QUICK_REFERENCE.md (all sections)
2. **Week 1:** Run `python scripts/usage_examples.py`
3. **Week 2:** Read DATABASE_DOCUMENTATION.md (sections 1–6)
4. **Week 2:** Study [ARCHITECTURE.md](ARCHITECTURE.md) diagrams
5. **Week 3:** Dive into [src/epic_preprocess.py](../src/epic_preprocess.py) code

---

## FAQ: Where Do I Find...?

| Question | Answer |
|----------|--------|
| "What are the feature columns?" | [DB_DOC §1.2](DATABASE_DOCUMENTATION.md#12-raw-csv-structure), [QUICK_REF](QUICK_REFERENCE.md#feature-definitions) |
| "How do I load an embryo?" | [QUICK_REF](QUICK_REFERENCE.md#minimal-working-example), [Example 1](../scripts/usage_examples.py) |
| "What's the shape of X?" | [DB_DOC §3.3](DATABASE_DOCUMENTATION.md#x-node-feature-tensor-n-d-t), [QUICK_REF § Dimensions](QUICK_REFERENCE.md#dimensions-at-a-glance) |
| "Why are some cells all zeros?" | [DB_DOC §3.3](DATABASE_DOCUMENTATION.md#masking), [QUICK_REF §Masking](QUICK_REFERENCE.md#masked-cells-growing-embryo) |
| "How are edges encoded?" | [DB_DOC §3.3](DATABASE_DOCUMENTATION.md#sparse-edge-lists-edge_src-edge_dst-edge_t), [ARCH](ARCHITECTURE.md#edge-construction-logic) |
| "What do lineage edges mean?" | [DB_DOC §4.2](DATABASE_DOCUMENTATION.md#biological-meaning-of-features--graphs), [ARCH § Lineage](ARCHITECTURE.md#lineage-edges-directed-ancestry) |
| "How do I find a cell's daughters?" | [QUICK_REF](QUICK_REFERENCE.md#cell-name--index-mapping), [Example 3](../scripts/usage_examples.py) |
| "How much memory do I need?" | [DB_DOC §5.2](DATABASE_DOCUMENTATION.md#52-compressed-npz-performance), [ARCH § Scalability](ARCHITECTURE.md#scalability) |
| "How do I run preprocessing?" | [DB_DOC §2.3](DATABASE_DOCUMENTATION.md#23-preprocessing-script), [README.md](../README.md) |
| "How do I validate my data?" | [DB_DOC §8](DATABASE_DOCUMENTATION.md#8-quality-control--validation), [Example 7](../scripts/usage_examples.py) |

---

## Key Concepts Glossary

### Core Terms

| Term | Definition | See Also |
|------|-----------|----------|
| **EPIC** | eMbryo Project Imaging Consortium — fluorescence microscopy dataset of C. elegans development | [DB_DOC §1.1](DATABASE_DOCUMENTATION.md#11-source-files) |
| **Embryo** | One complete developmental recording; stored as one NPZ file | All docs |
| **Cell** | Individual nucleus tracked through development; identified by C. elegans nomenclature (e.g., "ABal") | [DB_DOC §4.1](DATABASE_DOCUMENTATION.md#41-c-elegans-cell-nomenclature) |
| **Timepoint** | One frame of video; multiple rows per timepoint (one per cell) | [DB_DOC §1.2](DATABASE_DOCUMENTATION.md#12-raw-csv-structure) |
| **Node** | Synonym for cell when represented as graph node; indexed 0..N-1 | All docs |
| **Feature** | Measured property of a cell: x, y, z, size, blot (d=5 dimensions) | [QUICK_REF § Features](QUICK_REFERENCE.md#feature-definitions) |
| **Edge** | Connection between two cells; either spatial (proximity) or lineage (ancestry) | [ARCH § Edge Construction](ARCHITECTURE.md#edge-construction-logic) |
| **Spatial edge** | Undirected edge connecting physically close cells (< 20 μm) | [DB_DOC §2.2](DATABASE_DOCUMENTATION.md#step-3-build-spatial-edges-undirected) |
| **Lineage edge** | Directed edge from parent to daughter cell (inferred from naming) | [DB_DOC §2.2](DATABASE_DOCUMENTATION.md#step-4-build-lineage-edges-directed) |
| **Tensor** | Multi-dimensional NumPy array; e.g., X[N, d, T] | [DB_DOC §3](DATABASE_DOCUMENTATION.md#3-output-data-processed-npz-format) |
| **Sparse graph** | Edge list representation: (edge_src, edge_dst, edge_t) vs. dense adjacency A[N,N,T] | [DB_DOC §5.1](DATABASE_DOCUMENTATION.md#51-storage-comparison-dense-vs-sparse) |
| **alive_mask** | Boolean tensor [N, T] indicating when each cell is born and observed | [DB_DOC §3.3](DATABASE_DOCUMENTATION.md#alive_mask-birthalive-tracking-n-t) |
| **Masked cell** | Cell not yet born; has zero features and alive_mask[cell,t]=False | [QUICK_REF](QUICK_REFERENCE.md#masked-cells-growing-embryo) |

### Biological Terms

| Term | Definition | See Also |
|------|-----------|----------|
| **C. elegans** | Caenorhabditis elegans — nematode worm; standard model organism | [DB_DOC §4](DATABASE_DOCUMENTATION.md#4-biological-interpretation) |
| **Embryo** | Developmental stage from zygote (~1 cell) to ~700 cells | [DB_DOC §4.3](DATABASE_DOCUMENTATION.md#43-embryonic-development-timeline-example) |
| **Cell division** | Binary fission: one mother → two daughters; tracked via lineage tree | [DB_DOC §4.1](DATABASE_DOCUMENTATION.md#41-c-elegans-cell-nomenclature) |
| **Lineage** | Ancestry tree; graph of cell divisions from fertilized egg to final cells | [ARCH § Lineage Edges](ARCHITECTURE.md#lineage-edges-directed-ancestry) |
| **Cell naming** | Standard nomenclature encoding lineage: AB→ABa→ABal (last character = division history) | [DB_DOC §4.1](DATABASE_DOCUMENTATION.md#41-c-elegans-cell-nomenclature) |
| **Fluorescence** | "blot" feature; intensity of fluorescent marker for cell tracking | [DB_DOC §1.2](DATABASE_DOCUMENTATION.md#12-raw-csv-structure) |
| **Migration** | Cell movement tracked via (x, y, z) coordinates | [Example 5](../scripts/usage_examples.py) |

---

## Troubleshooting Guide

### Problem: "FileNotFoundError: No such file or directory"

**Likely cause:** You haven't run preprocessing yet

**Solution:**
```bash
cd d:\Github\Spatio-Temporal-Evolution
python scripts/preprocess_dataset.py --raw_dir dataset/raw --out dataset/processed/by_embryo
```

See [DB_DOC §2.3](DATABASE_DOCUMENTATION.md#23-preprocessing-script)

---

### Problem: "KeyError: 'X'"

**Likely cause:** Wrong file format; trying to load non-NPZ file

**Solution:**
```python
import numpy as np
# Correct: load from processed/by_embryo/
npz = np.load("dataset/processed/by_embryo/CD011605_5a_bright.npz", allow_pickle=True)
X = npz["X"]  # Now works
```

---

### Problem: Shape mismatch or "index out of range"

**Likely cause:** Different embryos have different N and T

**Solution:**
```python
# Always check shape per embryo
for npz_path in paths:
    npz = np.load(npz_path)
    N, d, T = npz["X"].shape
    print(f"{npz_path.name}: N={N}, T={T}")
```

See [Example 6](../scripts/usage_examples.py)

---

### Problem: Memory error loading all 260 embryos

**Likely cause:** Not enough RAM for all data simultaneously

**Solution:**
```python
# Load batch-wise
BATCH_SIZE = 10
for i in range(0, 260, BATCH_SIZE):
    batch = load_embryos(i, min(i+BATCH_SIZE, 260))
    # Process batch
    del batch  # Free memory
```

See [DB_DOC §5.2](DATABASE_DOCUMENTATION.md#52-compressed-npz-performance)

---

## Contact & Support

- **Code issues:** Check [DB_DOC §11](DATABASE_DOCUMENTATION.md#11-troubleshooting)
- **Data quality:** See [DB_DOC §8](DATABASE_DOCUMENTATION.md#8-quality-control--validation) for validation checklist
- **Biological questions:** Refer to [DB_DOC §4](DATABASE_DOCUMENTATION.md#4-biological-interpretation)

---

## Citation

If you use this preprocessed database, please cite:

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

---

**Documentation Index Version:** 1.0  
**Last Updated:** April 2026  
**Total Docs:** 4 files (~2000 lines + code examples)
