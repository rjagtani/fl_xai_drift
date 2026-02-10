"""
Evaluation metrics for drift diagnosis.

Computes Hits@K, Precision@K, Recall@K, and MRR for feature rankings.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import json
from pathlib import Path


@dataclass
class DiagnosticMetrics:
    """Metrics for evaluating a single diagnosis ranking."""
    
    method: str
    hits_at_k: float  # 1 if any drifted feature in top K
    precision_at_k: float  # |TopK ∩ S*| / K
    recall_at_k: float  # |TopK ∩ S*| / |S*|
    mrr: float  # 1 / (best rank of drifted feature)
    k: int
    top_k_features: List[int] = field(default_factory=list)
    drifted_in_top_k: List[int] = field(default_factory=list)


def compute_metrics(
    ranking: np.ndarray,
    ground_truth: Set[int],
    k: int = None,
) -> DiagnosticMetrics:
    """
    Compute diagnostic evaluation metrics for a feature ranking.
    
    Args:
        ranking: Array of feature indices sorted by importance (descending)
        ground_truth: Set of ground truth drifted feature indices
        k: Number of top features to consider (default: |ground_truth|)
    
    Returns:
        DiagnosticMetrics with all computed metrics
    """
    if k is None:
        k = len(ground_truth)
    
    if k == 0 or len(ground_truth) == 0:
        return DiagnosticMetrics(
            method='',
            hits_at_k=0.0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            k=k,
        )
    
    # Top K features
    top_k = set(ranking[:k].tolist())
    
    # Drifted features in top K
    drifted_in_top_k = top_k.intersection(ground_truth)
    
    # Hits@K: 1 if any drifted feature in top K
    hits_at_k = 1.0 if len(drifted_in_top_k) > 0 else 0.0
    
    # Precision@K: |TopK ∩ S*| / K
    precision_at_k = len(drifted_in_top_k) / k
    
    # Recall@K: |TopK ∩ S*| / |S*|
    recall_at_k = len(drifted_in_top_k) / len(ground_truth)
    
    # MRR: 1 / (best rank position of any drifted feature)
    # Rank positions are 1-indexed
    best_rank = None
    for pos, feature in enumerate(ranking):
        if feature in ground_truth:
            best_rank = pos + 1  # 1-indexed
            break
    
    mrr = 1.0 / best_rank if best_rank is not None else 0.0
    
    return DiagnosticMetrics(
        method='',  # To be set by caller
        hits_at_k=hits_at_k,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        k=k,
        top_k_features=list(ranking[:k]),
        drifted_in_top_k=list(drifted_in_top_k),
    )


def compute_all_metrics(
    rankings: Dict[str, np.ndarray],
    ground_truth: Set[int],
    k: int = None,
) -> Dict[str, DiagnosticMetrics]:
    """
    Compute metrics for all diagnosis methods.
    
    Args:
        rankings: Dictionary mapping method name to ranking array
        ground_truth: Set of ground truth drifted feature indices
        k: Number of top features to consider
    
    Returns:
        Dictionary mapping method name to DiagnosticMetrics
    """
    results = {}
    
    for method_name, ranking in rankings.items():
        metrics = compute_metrics(ranking, ground_truth, k)
        metrics.method = method_name
        results[method_name] = metrics
    
    return results


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""
    
    method: str
    hits_at_k_mean: float
    hits_at_k_std: float
    precision_at_k_mean: float
    precision_at_k_std: float
    recall_at_k_mean: float
    recall_at_k_std: float
    mrr_mean: float
    mrr_std: float
    n_runs: int


def aggregate_results(
    run_metrics: List[Dict[str, DiagnosticMetrics]],
) -> Dict[str, AggregatedMetrics]:
    """
    Aggregate metrics across multiple experimental runs.
    
    Args:
        run_metrics: List of metric dictionaries from each run
    
    Returns:
        Dictionary mapping method name to AggregatedMetrics
    """
    if not run_metrics:
        return {}
    
    # Collect metrics by method
    method_metrics: Dict[str, List[DiagnosticMetrics]] = {}
    
    for run_result in run_metrics:
        for method_name, metrics in run_result.items():
            if method_name not in method_metrics:
                method_metrics[method_name] = []
            method_metrics[method_name].append(metrics)
    
    # Aggregate
    aggregated = {}
    
    for method_name, metrics_list in method_metrics.items():
        n_runs = len(metrics_list)
        
        hits = [m.hits_at_k for m in metrics_list]
        precision = [m.precision_at_k for m in metrics_list]
        recall = [m.recall_at_k for m in metrics_list]
        mrr = [m.mrr for m in metrics_list]
        
        aggregated[method_name] = AggregatedMetrics(
            method=method_name,
            hits_at_k_mean=np.mean(hits),
            hits_at_k_std=np.std(hits),
            precision_at_k_mean=np.mean(precision),
            precision_at_k_std=np.std(precision),
            recall_at_k_mean=np.mean(recall),
            recall_at_k_std=np.std(recall),
            mrr_mean=np.mean(mrr),
            mrr_std=np.std(mrr),
            n_runs=n_runs,
        )
    
    return aggregated


def metrics_to_dict(metrics: DiagnosticMetrics) -> Dict[str, Any]:
    """Convert DiagnosticMetrics to dictionary for JSON serialization."""
    return {
        'method': metrics.method,
        'hits_at_k': int(metrics.hits_at_k) if metrics.hits_at_k is not None else None,
        'precision_at_k': float(metrics.precision_at_k) if metrics.precision_at_k is not None else None,
        'recall_at_k': float(metrics.recall_at_k) if metrics.recall_at_k is not None else None,
        'mrr': float(metrics.mrr) if metrics.mrr is not None else None,
        'k': int(metrics.k) if metrics.k is not None else None,
        'top_k_features': [int(f) for f in metrics.top_k_features] if metrics.top_k_features else [],
        'drifted_in_top_k': [int(f) for f in metrics.drifted_in_top_k] if metrics.drifted_in_top_k else [],
    }


def aggregated_to_dict(metrics: AggregatedMetrics) -> Dict[str, Any]:
    """Convert AggregatedMetrics to dictionary for JSON serialization."""
    return {
        'method': metrics.method,
        'hits_at_k': {'mean': metrics.hits_at_k_mean, 'std': metrics.hits_at_k_std},
        'precision_at_k': {'mean': metrics.precision_at_k_mean, 'std': metrics.precision_at_k_std},
        'recall_at_k': {'mean': metrics.recall_at_k_mean, 'std': metrics.recall_at_k_std},
        'mrr': {'mean': metrics.mrr_mean, 'std': metrics.mrr_std},
        'n_runs': metrics.n_runs,
    }


def save_metrics(
    metrics: Dict[str, DiagnosticMetrics],
    output_path: Path,
):
    """Save metrics to JSON file."""
    output_dict = {name: metrics_to_dict(m) for name, m in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)


def save_aggregated_metrics(
    metrics: Dict[str, AggregatedMetrics],
    output_path: Path,
):
    """Save aggregated metrics to JSON file."""
    output_dict = {name: aggregated_to_dict(m) for name, m in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)


def create_results_table(
    aggregated: Dict[str, AggregatedMetrics],
) -> str:
    """
    Create a formatted results table.
    
    Returns:
        String with formatted table
    """
    headers = ['Method', 'Hits@K', 'Precision@K', 'Recall@K', 'MRR']
    rows = []
    
    for method_name, metrics in sorted(aggregated.items()):
        rows.append([
            method_name,
            f"{metrics.hits_at_k_mean:.3f}±{metrics.hits_at_k_std:.3f}",
            f"{metrics.precision_at_k_mean:.3f}±{metrics.precision_at_k_std:.3f}",
            f"{metrics.recall_at_k_mean:.3f}±{metrics.recall_at_k_std:.3f}",
            f"{metrics.mrr_mean:.3f}±{metrics.mrr_std:.3f}",
        ])
    
    # Format table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    lines = []
    
    # Header
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append('-' * len(header_line))
    
    # Rows
    for row in rows:
        lines.append(' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    
    return '\n'.join(lines)
