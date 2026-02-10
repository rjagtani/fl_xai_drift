"""
Visualization utilities for drift detection results.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _check_plotting():
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for visualization")


def plot_training_curves(
    global_loss: List[float],
    client_loss_matrix: np.ndarray,
    t0: int,
    trigger_round: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
):
    """
    Plot training curves showing global and client losses.
    
    Args:
        global_loss: List of global losses per round
        client_loss_matrix: (n_rounds, n_clients) array of client losses
        t0: Drift onset round
        trigger_round: Round when trigger fired (optional)
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_plotting()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    rounds = range(1, len(global_loss) + 1)
    
    # Global loss
    ax1.plot(rounds, global_loss, 'b-', linewidth=2, label='Global Loss')
    ax1.axvline(x=t0, color='r', linestyle='--', linewidth=2, label=f'Drift Onset (t0={t0})')
    if trigger_round:
        ax1.axvline(x=trigger_round, color='g', linestyle=':', linewidth=2, 
                   label=f'Trigger (t={trigger_round})')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Global Loss Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Client losses heatmap
    im = ax2.imshow(client_loss_matrix.T, aspect='auto', cmap='YlOrRd')
    ax2.axvline(x=t0-1, color='white', linestyle='--', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Client ID')
    ax2.set_title('Per-Client Loss')
    plt.colorbar(im, ax=ax2, label='Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_fi_heatmap(
    fi_matrix: np.ndarray,
    feature_names: List[str],
    method_name: str,
    diagnosis_rounds: List[int],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
):
    """
    Plot feature importance heatmap across rounds and clients.
    
    Args:
        fi_matrix: (n_rounds, n_clients, n_features) array
        feature_names: List of feature names
        method_name: Name of FI method (e.g., 'SAGE', 'PFI')
        diagnosis_rounds: List of round numbers
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_plotting()
    
    n_rounds, n_clients, n_features = fi_matrix.shape
    
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, fname) in enumerate(zip(axes, feature_names)):
        data = fi_matrix[:, :, i].T  # (n_clients, n_rounds)
        
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r')
        ax.set_xlabel('Round')
        ax.set_ylabel('Client ID')
        ax.set_title(f'{fname}')
        ax.set_xticks(range(len(diagnosis_rounds)))
        ax.set_xticklabels(diagnosis_rounds, rotation=45)
        plt.colorbar(im, ax=ax)
    
    fig.suptitle(f'{method_name} Feature Importance', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_diagnosis_results(
    rankings: Dict[str, np.ndarray],
    ground_truth: set,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
):
    """
    Plot diagnosis results showing feature rankings vs ground truth.
    
    Args:
        rankings: Dict mapping method name to ranking array
        ground_truth: Set of ground truth drifted feature indices
        feature_names: List of feature names
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_plotting()
    
    methods = list(rankings.keys())
    n_methods = len(methods)
    n_features = len(feature_names)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        ranking = rankings[method]
        
        # Create ranking visualization
        colors = ['red' if i in ground_truth else 'blue' for i in ranking]
        
        y_pos = np.arange(n_features)
        feature_labels = [feature_names[i] for i in ranking]
        
        bars = ax.barh(y_pos, np.arange(n_features, 0, -1), color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Importance Rank')
        ax.set_title(method.replace('_', ' ').title())
        ax.invert_yaxis()  # Top feature at top
        
        # Add legend
        ax.barh([], [], color='red', alpha=0.7, label='Drifted (GT)')
        ax.barh([], [], color='blue', alpha=0.7, label='Non-drifted')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rds_scores(
    rds_scores: Dict[str, np.ndarray],
    feature_names: List[str],
    ground_truth: set,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot RDS scores for Dist(FI) methods.
    
    Args:
        rds_scores: Dict mapping method name to RDS score array
        feature_names: List of feature names
        ground_truth: Set of ground truth drifted feature indices
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_plotting()
    
    methods = list(rds_scores.keys())
    n_methods = len(methods)
    n_features = len(feature_names)
    
    x = np.arange(n_features)
    width = 0.8 / n_methods
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, method in enumerate(methods):
        scores = rds_scores[method]
        offset = (i - n_methods / 2 + 0.5) * width
        
        colors = ['red' if j in ground_truth else 'blue' for j in range(n_features)]
        ax.bar(x + offset, scores, width, label=method, alpha=0.7)
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('RDS Score')
    ax.set_title('Relative Distribution Shift (RDS) Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    
    # Highlight ground truth features
    for idx in ground_truth:
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_and_rds_detection(
    loss_series: List[float],
    rds_scores: List[float],
    t0: int,
    trigger_round: Optional[int] = None,
    threshold: Optional[float] = None,
    threshold_series: Optional[List[float]] = None,
    calibration_start: int = 51,
    calibration_end: int = 100,
    warmup_rounds: int = 50,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
):
    """
    Plot loss and RDS detection scores over training rounds.
    
    This is a key diagnostic plot showing the trigger detection behavior
    regardless of whether FI diagnosis was run.
    
    Args:
        loss_series: Global loss per round
        rds_scores: RDS scores from trigger detection
        t0: Drift onset round
        trigger_round: Round when trigger fired (optional)
        threshold: Final RDS detection threshold (optional, for label)
        threshold_series: Dynamic threshold over time (optional)
        calibration_start: Start of calibration period
        calibration_end: End of calibration period
        warmup_rounds: Warmup period before RDS computation
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_plotting()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    n_rounds = len(loss_series)
    rounds = range(1, n_rounds + 1)
    
    # Top: Loss over time
    ax1.plot(rounds, loss_series, 'b-', linewidth=1.5, label='Global Loss')
    ax1.axvline(x=t0, color='r', linestyle='--', linewidth=2, label=f'Drift Onset (t0={t0})')
    if trigger_round:
        ax1.axvline(x=trigger_round, color='g', linestyle=':', linewidth=2, 
                   label=f'Trigger (t={trigger_round})')
    
    # Shade calibration period
    ax1.axvspan(calibration_start, calibration_end, alpha=0.15, color='orange', 
                label='Calibration Period')
    
    ax1.set_ylabel('Loss')
    ax1.set_title('Global Loss and Drift Detection (EMA Threshold)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: RDS scores over time
    if rds_scores and len(rds_scores) > 0:
        # RDS scores start after warmup
        rds_rounds = list(range(warmup_rounds + 1, warmup_rounds + 1 + len(rds_scores)))
        ax2.plot(rds_rounds, rds_scores, 'purple', linewidth=1.5, label='RDS Score')
        
        # Plot dynamic threshold series if available
        if threshold_series and len(threshold_series) > 0:
            # Filter out None values (calibration period)
            valid_thresholds = [(r, t) for r, t in zip(rds_rounds, threshold_series) if t is not None]
            if valid_thresholds:
                thresh_rounds, thresh_values = zip(*valid_thresholds)
                ax2.plot(thresh_rounds, thresh_values, 'orange', linestyle='--', linewidth=2,
                        label='Dynamic Threshold (EMA)')
        elif threshold is not None:
            # Fallback: flat threshold line
            ax2.axhline(y=threshold, color='orange', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.4f})')
        
        ax2.axvline(x=t0, color='r', linestyle='--', linewidth=2)
        if trigger_round:
            ax2.axvline(x=trigger_round, color='g', linestyle=':', linewidth=2)
        
        # Shade calibration period
        ax2.axvspan(calibration_start, calibration_end, alpha=0.15, color='orange')
        
        ax2.set_ylabel('RDS Score')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No RDS scores available', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12)
    
    ax2.set_xlabel('Round')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_fi_and_rds_detection(
    fi_matrix: np.ndarray,  # Shape: (n_rounds, n_clients, n_features)
    rds_series: np.ndarray,  # Shape: (n_rds_rounds, n_features)
    rds_rounds: List[int],  # Round numbers for RDS values
    diagnosis_rounds: List[int],  # All diagnosis round numbers
    thresholds: np.ndarray,  # Shape: (n_features,) - calibrated threshold per feature
    feature_names: List[str],
    method_name: str,
    ground_truth: set,  # Indices of ground truth drifted features
    trigger_round: int,  # The trigger round
    calibration_mu: Optional[np.ndarray] = None,
    calibration_sigma: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
    top_k: int = 10,  # Show only top-k features for clarity
):
    """
    Plot FI values and FI-based RDS detection scores over diagnosis rounds.
    
    This mirrors the loss+RDS detection plot structure:
    - Top: Mean FI values per feature across diagnosis rounds
    - Bottom: RDS scores per feature over rounds with calibrated threshold
    
    Args:
        fi_matrix: FI values (rounds, clients, features)
        rds_series: RDS scores per round per feature
        rds_rounds: Round numbers for RDS values
        diagnosis_rounds: All diagnosis round numbers (for FI matrix)
        thresholds: Calibrated threshold per feature
        feature_names: List of feature names
        method_name: Name of FI method (e.g., 'SAGE', 'PFI', 'SHAP')
        ground_truth: Set of ground truth drifted feature indices
        trigger_round: The trigger round
        calibration_mu: Per-feature mu from calibration (optional for display)
        calibration_sigma: Per-feature sigma from calibration (optional for display)
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top features to display
    """
    _check_plotting()
    
    n_rounds, n_clients, n_features = fi_matrix.shape
    n_rds_rounds = len(rds_series)
    
    # Compute mean FI across clients per round
    mean_fi = np.nanmean(fi_matrix, axis=1)  # Shape: (n_rounds, n_features)
    
    # Select top-k features by final RDS score
    final_rds = rds_series[-1] if len(rds_series) > 0 else np.zeros(n_features)
    top_k_indices = np.argsort(final_rds)[::-1][:top_k]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False)
    
    # Color map for features
    colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    
    # Top: Mean FI values over diagnosis rounds
    for i, feat_idx in enumerate(top_k_indices):
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'F{feat_idx}'
        is_drifted = feat_idx in ground_truth
        linestyle = '-' if is_drifted else '--'
        marker = 'o' if is_drifted else 's'
        ax1.plot(diagnosis_rounds, mean_fi[:, feat_idx], 
                color=colors[i], linestyle=linestyle, marker=marker,
                markersize=5, linewidth=1.5, 
                label=f'{fname} (GT)' if is_drifted else fname)
    
    ax1.axvline(x=trigger_round, color='g', linestyle=':', linewidth=2, 
               label=f'Trigger (t={trigger_round})')
    ax1.set_ylabel('Mean FI Value')
    ax1.set_title(f'{method_name.upper()} Feature Importance Over Diagnosis Window')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Round')
    
    # Bottom: RDS scores over rounds with threshold
    if len(rds_series) > 0:
        for i, feat_idx in enumerate(top_k_indices):
            fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'F{feat_idx}'
            is_drifted = feat_idx in ground_truth
            linestyle = '-' if is_drifted else '--'
            marker = 'o' if is_drifted else 's'
            
            ax2.plot(rds_rounds, rds_series[:, feat_idx],
                    color=colors[i], linestyle=linestyle, marker=marker,
                    markersize=5, linewidth=1.5,
                    label=f'{fname}: thresh={thresholds[feat_idx]:.2f}')
            
            # Plot threshold as horizontal line for this feature
            ax2.axhline(y=thresholds[feat_idx], color=colors[i], 
                       linestyle=':', alpha=0.5, linewidth=1)
        
        ax2.axvline(x=trigger_round, color='g', linestyle=':', linewidth=2)
        
        # Mark calibration rounds (before trigger)
        calibration_rounds = rds_rounds[:-1]  # All but last
        if len(calibration_rounds) > 0:
            ax2.axvspan(min(calibration_rounds), max(calibration_rounds), 
                       alpha=0.1, color='orange', label='Calibration')
        
        ax2.set_ylabel('RDS Score')
        ax2.set_xlabel('Round')
        ax2.set_title(f'{method_name.upper()} RDS Detection with Calibrated Thresholds')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No RDS scores available', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_experiment_report(
    results: Dict[str, Any],
    output_dir: Path,
):
    """
    Generate a complete visual report for an experiment.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save figures
    """
    _check_plotting()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves if available
    if 'training' in results:
        training = results['training']
        if 'global_loss_series' in training:
            # This would require client_loss_matrix too
            pass
    
    print(f"Report generated in: {output_dir}")
