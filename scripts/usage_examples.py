#!/usr/bin/env python
"""
Usage Examples: EPIC Database Preprocessing & Analysis

Demonstrates how to:
1. Load processed NPZ tensors
2. Extract features and graphs
3. Analyze cell lineages
4. Compute statistics
5. Visualize results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_embryo(embryo_path: str | Path) -> dict:
    """Load a single preprocessed embryo NPZ file."""
    npz = np.load(embryo_path, allow_pickle=True)
    return {
        "X": npz["X"],
        "alive_mask": npz["alive_mask"],
        "edge_src": npz["edge_src"],
        "edge_dst": npz["edge_dst"],
        "edge_t": npz["edge_t"],
        "idx_to_cell": npz["idx_to_cell"],
        "t0": int(npz["t0"]),
        "T": int(npz["T"]),
        "source_file": str(npz["source_file"]),
    }


def get_cell_to_idx(idx_to_cell: np.ndarray) -> dict[str, int]:
    """Inverse mapping: cell name → index."""
    return {cell: idx for idx, cell in enumerate(idx_to_cell)}


# ============================================================================
# Example 1: Load & inspect a single embryo
# ============================================================================

def example_1_load_and_inspect():
    """Load one embryo and print basic statistics."""
    print("\n" + "="*80)
    print("Example 1: Load & Inspect Embryo")
    print("="*80)
    
    embryo_path = Path("dataset/processed/by_embryo/CD011605_5a_bright.npz")
    if not embryo_path.exists():
        print(f"⚠ File not found: {embryo_path}")
        return
    
    embryo = load_embryo(embryo_path)
    
    X = embryo["X"]
    alive_mask = embryo["alive_mask"]
    idx_to_cell = embryo["idx_to_cell"]
    
    N, d, T = X.shape
    E = len(embryo["edge_src"])
    
    print(f"\nEmbryofile: {embryo_path.name}")
    print(f"Original source: {embryo['source_file']}")
    print(f"\nDimensions:")
    print(f"  Cells (N):           {N}")
    print(f"  Features (d):        {d}")
    print(f"  Timepoints (T):      {T}")
    print(f"  Total edges (E):     {E}")
    
    # Birth times
    birth_times = alive_mask.argmax(axis=1)  # First True index per cell
    print(f"\nCell birth times:")
    print(f"  Earliest: t={birth_times.min()} (cell: {idx_to_cell[birth_times.argmin()]})")
    print(f"  Latest:   t={birth_times.max()}")
    
    # Lifespan
    lifespans = alive_mask.sum(axis=1)
    print(f"\nCell lifespans (observed timepoints):")
    print(f"  Mean:     {lifespans[lifespans > 0].mean():.1f}")
    print(f"  Max:      {lifespans.max()}")
    print(f"  Min:      {lifespans[lifespans > 0].min()}")
    
    # Feature statistics (only for alive cells)
    feature_names = ["x (px)", "y (px)", "z (μm)", "size (AU)", "blot (AU)"]
    print(f"\nFeature statistics (alive cells only):")
    for d_idx, name in enumerate(feature_names):
        X_feat = X[:, d_idx, :][alive_mask]  # Extract alive measurements
        if len(X_feat) > 0:
            print(f"  {name:15} → mean={X_feat.mean():.1f}, "
                  f"std={X_feat.std():.1f}, "
                  f"min={X_feat.min():.1f}, "
                  f"max={X_feat.max():.1f}")


# ============================================================================
# Example 2: Extract features at a specific timepoint
# ============================================================================

def example_2_features_at_timepoint():
    """Extract node features at time t and analyze active cells."""
    print("\n" + "="*80)
    print("Example 2: Features at Timepoint")
    print("="*80)
    
    embryo_path = Path("dataset/processed/by_embryo/CD011605_5a_bright.npz")
    if not embryo_path.exists():
        print(f"⚠ File not found: {embryo_path}")
        return
    
    embryo = load_embryo(embryo_path)
    t_idx = 100  # Timepoint index (0-indexed)
    
    X = embryo["X"]
    alive_mask = embryo["alive_mask"]
    idx_to_cell = embryo["idx_to_cell"]
    
    # All cells' features at this time
    X_t = X[:, :, t_idx]  # (N, 5)
    alive_t = alive_mask[:, t_idx]  # (N,)
    
    n_alive = alive_t.sum()
    print(f"\nAt timepoint {t_idx}:")
    print(f"  Total cells:     {len(X_t)}")
    print(f"  Alive cells:     {n_alive}")
    
    # Extract only alive cells
    alive_indices = np.where(alive_t)[0]
    X_active = X[alive_indices, :, t_idx]  # (M, 5)
    
    print(f"\nActive cell features (mean):")
    feature_names = ["x", "y", "z", "size", "blot"]
    for d_idx, name in enumerate(feature_names):
        print(f"  {name:8} → {X_active[:, d_idx].mean():.2f} ± {X_active[:, d_idx].std():.2f}")
    
    # Top 10 brightest cells
    blot_idx = 4
    blot_values = X_active[:, blot_idx]
    top_indices = blot_values.argsort()[-10:][::-1]
    
    print(f"\nTop 10 brightest cells (high 'blot' feature):")
    for rank, local_idx in enumerate(top_indices, 1):
        global_idx = alive_indices[local_idx]
        cell_name = idx_to_cell[global_idx]
        blot = X_active[local_idx, blot_idx]
        print(f"  {rank:2}. {cell_name:20} → blot={blot:.0f}")


# ============================================================================
# Example 3: Analyze cell lineage
# ============================================================================

def example_3_lineage_analysis():
    """Reconstruct cell lineage tree from naming convention."""
    print("\n" + "="*80)
    print("Example 3: Lineage Analysis")
    print("="*80)
    
    embryo_path = Path("dataset/processed/by_embryo/CD011605_5a_bright.npz")
    if not embryo_path.exists():
        print(f"⚠ File not found: {embryo_path}")
        return
    
    embryo = load_embryo(embryo_path)
    idx_to_cell = embryo["idx_to_cell"]
    
    # Build parent-daughter map from cell names
    lineage = defaultdict(list)
    for cell_name in idx_to_cell:
        if len(cell_name) > 1 and cell_name[-1].isalpha():
            parent = cell_name[:-1]
            if parent in set(idx_to_cell):
                lineage[parent].append(cell_name)
    
    print(f"\nReconstru cted lineage tree ({len(lineage)} mothers):")
    print(f"\nExample hierarchies:")
    
    # Print sample lineage paths
    root_cells = ["AB", "P1", "EMS", "MS", "C", "D", "Z2", "Z3"]
    for root in root_cells:
        if root not in lineage:
            continue
        
        print(f"\n{root} divides into:")
        for daughter in sorted(lineage[root]):
            print(f"  → {daughter}")
            # Third-generation granddaughters
            if daughter in lineage:
                for granddaughter in sorted(lineage[daughter])[:3]:
                    print(f"      → {granddaughter}")
                if len(lineage[daughter]) > 3:
                    print(f"      → ... +{len(lineage[daughter])-3} more")


# ============================================================================
# Example 4: Build spatial graph at timepoint
# ============================================================================

def example_4_spatial_graph():
    """Extract spatial edges at a specific time."""
    print("\n" + "="*80)
    print("Example 4: Spatial Graph at Timepoint")
    print("="*80)
    
    embryo_path = Path("dataset/processed/by_embryo/CD011605_5a_bright.npz")
    if not embryo_path.exists():
        print(f"⚠ File not found: {embryo_path}")
        return
    
    embryo = load_embryo(embryo_path)
    t_idx = 100
    
    N = embryo["X"].shape[0]
    edge_src = embryo["edge_src"]
    edge_dst = embryo["edge_dst"]
    edge_t = embryo["edge_t"]
    idx_to_cell = embryo["idx_to_cell"]
    
    # Filter edges at time t_idx
    mask = edge_t == t_idx
    edges_at_t = (edge_src[mask], edge_dst[mask])
    
    print(f"\nAt timepoint {t_idx}:")
    print(f"  Total edges: {mask.sum()}")
    
    # Build adjacency matrix
    A_t = np.zeros((N, N), dtype=bool)
    A_t[edges_at_t[0], edges_at_t[1]] = True
    
    # Compute graph statistics
    degree = A_t.sum(axis=1)
    print(f"\nGraph statistics:")
    print(f"  Average degree:  {degree.mean():.2f}")
    print(f"  Max degree:      {degree.max()}")
    print(f"  Density:         {mask.sum() / (N*N):.6f}")
    
    # Top-degree nodes (high local connectivity)
    top_nodes = np.argsort(degree)[-5:][::-1]
    print(f"\nTop 5 most connected cells:")
    for rank, node_idx in enumerate(top_nodes, 1):
        cell_name = idx_to_cell[node_idx]
        deg = degree[node_idx]
        print(f"  {rank}. {cell_name:20} → degree {deg}")


# ============================================================================
# Example 5: Cell trajectory (spacetime)
# ============================================================================

def example_5_cell_trajectory():
    """Follow a single cell's 3D position over time."""
    print("\n" + "="*80)
    print("Example 5: Single Cell Trajectory")
    print("="*80)
    
    embryo_path = Path("dataset/processed/by_embryo/CD011605_5a_bright.npz")
    if not embryo_path.exists():
        print(f"⚠ File not found: {embryo_path}")
        return
    
    embryo = load_embryo(embryo_path)
    
    # Track cell "ABal"
    target_cell = "ABal"
    idx_to_cell = embryo["idx_to_cell"]
    cell_to_idx = get_cell_to_idx(idx_to_cell)
    
    if target_cell not in cell_to_idx:
        print(f"\n⚠ Cell '{target_cell}' not found in this embryo")
        return
    
    cell_idx = cell_to_idx[target_cell]
    X = embryo["X"]
    alive_mask = embryo["alive_mask"]
    
    # Extract trajectory
    trajectory = []
    for t_idx in range(X.shape[2]):
        if alive_mask[cell_idx, t_idx]:
            x, y, z, size, blot = X[cell_idx, :, t_idx]
            trajectory.append({
                "time": t_idx,
                "x": x,
                "y": y,
                "z": z,
                "size": size,
                "blot": blot,
            })
    
    print(f"\nTrajectory of cell: {target_cell}")
    print(f"  Active for {len(trajectory)} timepoints")
    
    if len(trajectory) > 0:
        traj_df = pd.DataFrame(trajectory)
        print(f"\n{target_cell} movement statistics:")
        print(f"  Time range: {traj_df['time'].min()} → {traj_df['time'].max()}")
        
        # Displacement
        dx = traj_df["x"].iloc[-1] - traj_df["x"].iloc[0]
        dy = traj_df["y"].iloc[-1] - traj_df["y"].iloc[0]
        dz = traj_df["z"].iloc[-1] - traj_df["z"].iloc[0]
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"  Displacement (origin → final): {displacement:.1f} μm")
        print(f"    Δx={dx:.1f}, Δy={dy:.1f}, Δz={dz:.1f}")
        
        # Size changes (cell growth)
        print(f"  Size change: {traj_df['size'].iloc[0]:.0f} → {traj_df['size'].iloc[-1]:.0f}")
        print(f"  Blot (identity): {traj_df['blot'].iloc[0]:.0f} → {traj_df['blot'].iloc[-1]:.0f}")


# ============================================================================
# Example 6: Batch statistics across all embryos
# ============================================================================

def example_6_batch_statistics():
    """Load all embryos and compute summary statistics."""
    print("\n" + "="*80)
    print("Example 6: Batch Statistics (All Embryos)")
    print("="*80)
    
    base_dir = Path("dataset/processed/by_embryo")
    npz_files = sorted(base_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"⚠ No NPZ files found in {base_dir}")
        return
    
    stats = {
        "n_embryos": 0,
        "n_cells": [],
        "n_timepoints": [],
        "n_edges": [],
    }
    
    print(f"\nProcessing {len(npz_files)} embryos...")
    for npz_path in npz_files[:10]:  # Limit to first 10 for speed
        embryo = load_embryo(npz_path)
        N, d, T = embryo["X"].shape
        E = len(embryo["edge_src"])
        
        stats["n_embryos"] += 1
        stats["n_cells"].append(N)
        stats["n_timepoints"].append(T)
        stats["n_edges"].append(E)
    
    # Summarize
    print(f"\nBatch statistics (10 embryos shown):")
    print(f"  Total embryos scanned:    {stats['n_embryos']}")
    print(f"  Cells (N):")
    print(f"    Mean:     {np.mean(stats['n_cells']):.0f}")
    print(f"    Std:      {np.std(stats['n_cells']):.0f}")
    print(f"    Range:    {np.min(stats['n_cells'])} – {np.max(stats['n_cells'])}")
    
    print(f"  Timepoints (T):")
    print(f"    Mean:     {np.mean(stats['n_timepoints']):.0f}")
    print(f"    Range:    {np.min(stats['n_timepoints'])} – {np.max(stats['n_timepoints'])}")
    
    print(f"  Edges (E):")
    print(f"    Mean:     {np.mean(stats['n_edges']):.0f}")
    print(f"    Range:    {np.min(stats['n_edges'])} – {np.max(stats['n_edges'])}")


# ============================================================================
# Example 7: Data validation (QC checks)
# ============================================================================

def example_7_validate_embryo():
    """Run quality checks on an embryo's tensors."""
    print("\n" + "="*80)
    print("Example 7: Data Validation (QC)")
    print("="*80)
    
    embryo_path = Path("dataset/processed/by_embryo/CD011605_5a_bright.npz")
    if not embryo_path.exists():
        print(f"⚠ File not found: {embryo_path}")
        return
    
    embryo = load_embryo(embryo_path)
    
    X = embryo["X"]
    alive_mask = embryo["alive_mask"]
    edge_src = embryo["edge_src"]
    edge_dst = embryo["edge_dst"]
    edge_t = embryo["edge_t"]
    idx_to_cell = embryo["idx_to_cell"]
    
    N, d, T = X.shape
    
    checks = []
    
    # Check 1: Shape consistency
    try:
        assert alive_mask.shape == (N, T), f"alive_mask shape mismatch"
        checks.append(("✓", "alive_mask shape consistent"))
    except AssertionError as e:
        checks.append(("✗", str(e)))
    
    # Check 2: Edge indices valid
    try:
        assert edge_src.max() < N and edge_src.min() >= 0
        assert edge_dst.max() < N and edge_dst.min() >= 0
        assert edge_t.max() < T and edge_t.min() >= 0
        checks.append(("✓", "Edge indices in valid range"))
    except AssertionError:
        checks.append(("✗", "Edge indices out of range"))
    
    # Check 3: Unborn cells are zero
    try:
        unborn_mask = ~alive_mask
        unborn_X = X[unborn_mask]
        assert (unborn_X == 0).all(), "Some unborn cells have non-zero features"
        checks.append(("✓", "Unborn cells all zero"))
    except AssertionError as e:
        checks.append(("✗", str(e)))
    
    # Check 4: Cell names unique
    try:
        assert len(idx_to_cell) == N
        assert len(set(idx_to_cell)) == N, "Duplicate cell names"
        checks.append(("✓", "Cell names unique"))
    except AssertionError as e:
        checks.append(("✗", str(e)))
    
    # Check 5: Features in reasonable range
    try:
        X_alive = X[alive_mask]
        assert (X_alive >= 0).all(), "Negative coordinates found"
        assert X_alive[:, :3].max() < 1000, "Coordinates too large"
        checks.append(("✓", "Features in reasonable ranges"))
    except AssertionError as e:
        checks.append(("✗", str(e)))
    
    # Print results
    print(f"\nValidation checks for {embryo_path.name}:")
    for status, message in checks:
        print(f"  {status} {message}")
    
    passed = sum(1 for s, _ in checks if s == "✓")
    total = len(checks)
    print(f"\nResult: {passed}/{total} checks passed")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EPIC Database Usage Examples")
    print("="*80)
    
    example_1_load_and_inspect()
    example_2_features_at_timepoint()
    example_3_lineage_analysis()
    example_4_spatial_graph()
    example_5_cell_trajectory()
    example_6_batch_statistics()
    example_7_validate_embryo()
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80 + "\n")
