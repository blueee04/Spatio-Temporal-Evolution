## Slide 1 — EPIC → Spatio‑Temporal Graph Tensors (What & Why)

**Big idea:** One EPIC CSV already contains *time-varying cell features* + enough naming structure to infer *directed lineage edges*.

**Picture (full slide):** Pipeline overview diagram  
`Raw EPIC CSV → (X, alive_mask, edges) → ESTGEL / GNN training`

**Tiny caption (1 line):** We turn a single embryo movie table into time-aligned node features and a dynamic graph.

**1–2 bullets (optional, small):**
- Fixed tensor shapes per embryo: \(X \in \mathbb{R}^{N \times d \times T}\), \(d=5\)
- Dynamic graph stored sparsely as edges across time

---

## Slide 2 — Inputs: EPIC file format (What we read)

**Picture (full slide):** Screenshot-style table snippet of the CSV header + 3 rows  
Show columns: `cell,time,x,y,z,size,blot` (and optionally other intensity columns)

**Callouts on the picture (3 labels):**
- **cell**: biological identity (e.g., `ABa`, `ABal`, `ABala`)
- **time**: discrete time point (1…T)
- **features**: \(x,y,z\) position + `size` + `blot` (signal)

**Tiny caption:** Each row is one cell at one time.

---

## Slide 3 — Output 1: Feature tensor + “growing node” mask

**Picture (full slide):** Heatmap / tensor cartoon  
Left: grid of \(N\) cells × \(T\) time, with colored blocks where cells exist  
Right: a 3D block labeled \(X[N, d, T]\) where \(d=\{x,y,z,size,blot\}\)

**Key labels on the graphic:**
- **alive_mask[N,T]** = 1 if cell is present at that time
- **Unborn cells** → all-zero feature vector (natural masking)

**1 bullet (optional):**
- This satisfies “fixed \(N\)” while representing biological growth.

---

## Slide 4 — Output 2: Dynamic graph edges (Spatial + Lineage)

**Picture (full slide):** Two-layer graph illustration at one time \(t\)
- **Spatial edges (undirected):** connect nearby cells if distance < threshold (3D proximity)
- **Lineage edges (directed):** parent → daughter inferred from names  
Example label: `ABa → ABal` (strip last alphabetic character)

**Small box (bottom-right):** Why sparse edges?
- Dense \(A \in \mathbb{R}^{N \times N \times T}\) is huge
- Store edges as `edge_src, edge_dst, edge_t` (efficient)

**Tiny caption:** Same biology, memory-safe representation.

