"""
Main experiment runner that orchestrates the complete pipeline.

Pipeline stages:
1. Data generation
2. FL training with checkpoint saving
3. Drift trigger detection (Page-Hinkley on global loss)
4. Feature importance computation (SAGE, PFI, SHAP)
5. Diagnosis (Dist(FI), Mean(FI)+PH)
6. Metrics evaluation
7. Results aggregation
"""

from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import numpy as np
import json
import time

from ..config import ExperimentConfig, create_experiment_configs
from ..fl_trainer import FLTrainer
from ..triggers import PageHinkleyDetector, MultiMethodDetector, FLBaselineMultiDetector
from ..diagnosis import DiagnosisEngine
from ..metrics import compute_all_metrics, aggregate_results, save_metrics, save_aggregated_metrics, create_results_table
from ..utils.visualization import (
    plot_training_curves, plot_fi_heatmap, plot_diagnosis_results, 
    plot_rds_scores, plot_loss_and_rds_detection, plot_fi_and_rds_detection
)


class ExperimentRunner:
    """
    Main runner for drift detection experiments.
    
    Executes the complete pipeline:
    1. Train FL model
    2. Detect drift trigger
    3. Compute FI for diagnosis window
    4. Run diagnosis methods
    5. Evaluate metrics
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self.config = config
        self.trainer: Optional[FLTrainer] = None
        self.trigger_result: Optional[Any] = None
        self.fl_baseline_results: Optional[Dict[str, Any]] = None
        self.diagnosis_results: Optional[Dict[str, Any]] = None
        self.metrics_results: Optional[Dict[str, Any]] = None
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run FL training phase.
        
        Returns:
            Training summary with timing
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Federated Learning Training")
        print("=" * 60)
        
        stage_start = time.time()
        self.trainer = FLTrainer(self.config)
        summary = self.trainer.train()
        stage_elapsed = time.time() - stage_start
        
        summary['stage_time_seconds'] = stage_elapsed
        print(f"  Stage 1 time: {stage_elapsed:.1f}s")
        
        return summary
    
    def run_trigger_detection(self) -> Dict[str, Any]:
        """
        Run drift trigger detection using 3 methods: RDS, Page-Hinkley, CUSUM.
        
        Returns:
            Trigger detection results with timing
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Drift Trigger Detection (RDS + PH + CUSUM)")
        print("=" * 60)
        
        stage_start = time.time()
        
        if self.trainer is None:
            raise RuntimeError("Must run training before trigger detection")
        
        # Get loss data
        loss_series = np.array(self.trainer.global_loss_series)
        loss_matrix = np.load(self.config.logs_dir / "client_loss_matrix.npy")
        
        t0 = self.config.drift.t0
        
        # Create multi-method detector using config
        trigger_cfg = self.config.trigger
        detector = MultiMethodDetector(
            warmup_rounds=trigger_cfg.warmup_rounds,
            calibration_start=trigger_cfg.calibration_start_round,
            calibration_end=trigger_cfg.calibration_end_round,
            n_rounds=self.config.fl.n_rounds,
            rds_window=trigger_cfg.rds_window,
            rds_alpha=trigger_cfg.rds_alpha,
            confirm_consecutive=trigger_cfg.confirm_consecutive,
            min_instances=trigger_cfg.min_instances,
            cusum_k_ref=trigger_cfg.cusum_k_ref,
            cusum_h=trigger_cfg.cusum_h,
            fwer_p=trigger_cfg.fwer_p,
        )
        
        # Run all 3 detection methods
        results = detector.detect(loss_matrix, loss_series, t0)
        
        # Log results for each method
        print(f"  Drift onset (t0): round {t0}")
        print()
        
        for method_name, result in results.items():
            status = "TRIGGERED" if result.triggered else "not triggered"
            print(f"  {method_name.upper()}: {status}")
            if result.triggered:
                print(f"    Trigger round: {result.trigger_round}")
                print(f"    Detection delay: {result.detection_delay} rounds")
            if result.score is not None:
                print(f"    Score: {result.score:.4f}")
        
        # Prefer RDS trigger if it triggered at or after calibration
        # Filter out false positives (triggers before t0) for other methods
        calibration_end = self.config.trigger.calibration_end_round
        
        rds_result = results['rds']
        if rds_result.triggered and rds_result.trigger_round >= calibration_end:
            # RDS triggered after calibration - use it
            self.trigger_result = rds_result
            triggered = True
            trigger_round = rds_result.trigger_round
            detection_delay = rds_result.detection_delay
            trigger_method = 'rds'
        else:
            # Check other methods for valid triggers (after t0)
            valid_triggers = [
                r for r in results.values()
                if r.triggered and r.trigger_round is not None and r.trigger_round >= t0
            ]
            if valid_triggers:
                first_valid = min(valid_triggers, key=lambda r: r.trigger_round)
                self.trigger_result = first_valid
                triggered = True
                trigger_round = first_valid.trigger_round
                detection_delay = first_valid.detection_delay
                trigger_method = first_valid.method
            else:
                # No valid trigger - use RDS result (may have max score round)
                self.trigger_result = rds_result
                triggered = False
                trigger_round = rds_result.trigger_round
                detection_delay = rds_result.detection_delay
                trigger_method = None
        
        print()
        if triggered:
            print(f"  => Selected trigger: {trigger_method.upper()} at round {trigger_round} (delay: {detection_delay})")
        else:
            print("  => No valid trigger detected (all triggers before t0 are false positives)")
        
        # Save trigger results
        trigger_results = {
            'triggered': triggered,
            'trigger_round': trigger_round,
            'detection_delay': detection_delay,
            'trigger_method': trigger_method,
            't0': t0,
            'methods': {
                name: {
                    'triggered': r.triggered,
                    'trigger_round': r.trigger_round,
                    'detection_delay': r.detection_delay,
                    'score': r.score,
                    'threshold': r.threshold,
                }
                for name, r in results.items()
            },
            'rds_scores': results['rds'].all_scores,
            'rds_threshold_series': results['rds'].threshold_series,
            'loss_series': loss_series.tolist(),
        }
        
        with open(self.config.logs_dir / "trigger_results.json", 'w') as f:
            json.dump({k: v for k, v in trigger_results.items() 
                      if k not in ['loss_series', 'rds_scores']}, f, indent=2)
        
        # Save RDS scores for plotting
        if results['rds'].all_scores:
            np.save(self.config.logs_dir / "rds_scores.npy", np.array(results['rds'].all_scores))
        
        stage_elapsed = time.time() - stage_start
        trigger_results['stage_time_seconds'] = stage_elapsed
        print(f"  Stage 2 time: {stage_elapsed:.1f}s")
        
        return trigger_results
    
    def run_fl_baselines(self) -> Dict[str, Any]:
        """
        Run FL baseline detectors: FLASH, CDA-FedAvg, Manias (PCA+KMeans).
        
        Uses ORIGINAL signals:
        - FLASH: aggregated weight updates (gradient disparity)
        - CDA-FedAvg: per-client confidence scores (max posterior probability)
        - Manias: per-client weight updates (PCA + KMeans)
        
        System-level evaluation:
        - FP if ANY client triggers before t0
        - Success if ALL drifted clients trigger at or after t0
        
        Returns:
            FL baseline detection results with timing
        """
        print("\n" + "=" * 60)
        print("STAGE 2b: FL Baseline Detection (FLASH, CDA-FedAvg, Manias)")
        print("  Using ORIGINAL signals: weight updates, confidence scores")
        print("=" * 60)
        
        stage_start = time.time()
        
        if self.trainer is None:
            raise RuntimeError("Must run training before FL baseline detection")
        
        # Ground truth from config
        t0 = self.config.drift.t0
        n_clients = self.config.fl.n_clients
        drifted_proportion = self.config.drift.drifted_client_proportion
        n_drifted = int(n_clients * drifted_proportion)
        ground_truth_clients = set(range(n_drifted))
        
        print(f"  Ground truth drifted clients: {ground_truth_clients}")
        print(f"  Drift onset (t0): round {t0}")
        
        # Get FL baseline signals from trainer
        confidence_matrix = self.trainer.client_confidence_matrix  # (n_rounds, n_clients)
        
        # Get aggregated weight updates for FLASH
        aggregated_updates = None
        if self.trainer.aggregated_weight_updates:
            flattened = []
            for update in self.trainer.aggregated_weight_updates:
                flat = np.concatenate([u.flatten() for u in update])
                flattened.append(flat)
            aggregated_updates = np.array(flattened)
        
        # Get per-client weight updates for Manias
        client_weight_updates = None
        if self.trainer.client_weight_updates:
            n_rounds = len(self.trainer.client_weight_updates)
            n_params = None
            for round_updates in self.trainer.client_weight_updates:
                if round_updates:
                    first_update = next(iter(round_updates.values()))
                    n_params = sum(u.size for u in first_update)
                    break
            
            if n_params:
                client_weight_updates = np.full((n_rounds, n_clients, n_params), np.nan)
                for r, round_updates in enumerate(self.trainer.client_weight_updates):
                    for client_id, update in round_updates.items():
                        flat = np.concatenate([u.flatten() for u in update])
                        client_weight_updates[r, client_id, :] = flat
        
        # Get calibration parameters
        trigger_cfg = self.config.trigger
        calibration_start = trigger_cfg.calibration_start_round
        calibration_end = trigger_cfg.calibration_end_round
        confirm_consecutive = trigger_cfg.confirm_consecutive
        
        # Create FL baseline detector
        fl_baseline = FLBaselineMultiDetector(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            confirm_consecutive=confirm_consecutive,
            alpha=3.0,
        )
        
        # Run detection with original signals
        results = fl_baseline.detect(
            aggregated_updates=aggregated_updates,
            confidence_matrix=confidence_matrix,
            client_weight_updates=client_weight_updates,
            drifted_clients=ground_truth_clients,
            t0=t0,
        )
        
        # Log results
        print()
        for method_name, system_result in results.items():
            status = "SUCCESS" if system_result.triggered else "NOT TRIGGERED"
            fp_status = " (HAS FP)" if system_result.has_false_positive else ""
            print(f"  {method_name.upper()}: {status}{fp_status}")
            print(f"    System trigger round: {system_result.trigger_round}")
            if system_result.detection_delay is not None:
                print(f"    Detection delay: {system_result.detection_delay} rounds")
            if system_result.has_false_positive:
                print(f"    False positive clients: {system_result.false_positive_clients}")
            if system_result.client_results:
                n_triggered = sum(1 for c in system_result.client_results.values() if c.triggered)
                print(f"    Clients triggered: {n_triggered}/{len(system_result.client_results)}")
        
        # Compute trigger metrics for each baseline
        baseline_trigger_metrics = {}
        for method_name, system_result in results.items():
            triggered = system_result.triggered
            has_fp = system_result.has_false_positive
            
            # TP: success (all drifted detected at/after t0, no FP)
            # FP: any client triggered before t0
            # FN: not all drifted detected and no FP
            if triggered:
                tp, fp, fn = 1, 0, 0
            elif has_fp:
                tp, fp, fn = 0, 1, 0
            else:
                tp, fp, fn = 0, 0, 1
            
            baseline_trigger_metrics[method_name] = {
                'triggered': triggered,
                'trigger_round': system_result.trigger_round,
                'detection_delay': system_result.detection_delay,
                'has_false_positive': has_fp,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            }
        
        # Save results
        fl_baseline_results = {
            't0': t0,
            'ground_truth_clients': list(ground_truth_clients),
            'methods': {},
        }
        
        for method_name, system_result in results.items():
            client_results_dict = {}
            if system_result.client_results:
                for c_id, c_result in system_result.client_results.items():
                    client_results_dict[str(c_id)] = {
                        'triggered': c_result.triggered,
                        'trigger_round': c_result.trigger_round,
                    }
            
            fl_baseline_results['methods'][method_name] = {
                'system_triggered': system_result.triggered,
                'system_trigger_round': system_result.trigger_round,
                'has_false_positive': system_result.has_false_positive,
                'false_positive_clients': list(system_result.false_positive_clients) if system_result.false_positive_clients else [],
                'client_results': client_results_dict,
                'trigger_metrics': baseline_trigger_metrics[method_name],
            }
        
        with open(self.config.logs_dir / "fl_baseline_results.json", 'w') as f:
            json.dump(fl_baseline_results, f, indent=2)
        
        stage_elapsed = time.time() - stage_start
        fl_baseline_results['stage_time_seconds'] = stage_elapsed
        print(f"\n  Stage 2b time: {stage_elapsed:.1f}s")
        
        self.fl_baseline_results = fl_baseline_results
        return fl_baseline_results
    
    def run_diagnosis(self, trigger_round: int = None) -> Dict[str, Any]:
        """
        Run diagnosis phase.
        
        Args:
            trigger_round: Round to use for diagnosis (default: from trigger detection)
        
        Returns:
            Diagnosis results with timing
        """
        print("\n" + "=" * 60)
        print("STAGE 3: Diagnosis")
        print("=" * 60)
        
        stage_start = time.time()
        
        if self.trainer is None:
            raise RuntimeError("Must run training before diagnosis")
        
        # Determine trigger round
        if trigger_round is None:
            if self.trigger_result is not None and self.trigger_result.triggered:
                trigger_round = self.trigger_result.trigger_round
            else:
                # Use last round if no trigger
                trigger_round = self.config.fl.n_rounds
                print(f"  No trigger detected, using final round: {trigger_round}")
        
        print(f"  Diagnosis trigger round: {trigger_round}")
        
        # Create diagnosis engine
        engine = DiagnosisEngine(self.config, self.trainer)
        
        # Run diagnosis
        self.diagnosis_results = engine.run_diagnosis(trigger_round)
        
        stage_elapsed = time.time() - stage_start
        self.diagnosis_results['stage_time_seconds'] = stage_elapsed
        print(f"  Stage 3 time: {stage_elapsed:.1f}s")
        
        return self.diagnosis_results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run metrics evaluation.
        
        Returns:
            Evaluation results with timing
        """
        print("\n" + "=" * 60)
        print("STAGE 4: Metrics Evaluation")
        print("=" * 60)
        
        stage_start = time.time()
        
        if self.diagnosis_results is None:
            raise RuntimeError("Must run diagnosis before evaluation")
        
        # Get ground truth drifted features
        ground_truth = self.trainer.get_drifted_feature_indices()
        
        # Override with config if specified
        if self.config.drift.drifted_features:
            ground_truth = self.config.drift.drifted_features
        
        print(f"  Ground truth drifted features: {ground_truth}")
        
        # Determine K
        k = self.config.metrics.k
        if self.config.metrics.use_ground_truth_k:
            k = len(ground_truth)
        print(f"  Using K = {k}")
        
        # Get rankings from diagnosis
        rankings = {}
        
        for name, result in self.diagnosis_results['dist_fi'].items():
            rankings[name] = result.feature_ranking
        
        for name, result in self.diagnosis_results['mean_fi_ph'].items():
            rankings[name] = result.feature_ranking
        
        # Compute metrics
        metrics = compute_all_metrics(rankings, ground_truth, k)
        
        self.metrics_results = metrics
        
        # Print results
        print("\n  Results:")
        for name, m in sorted(metrics.items()):
            print(f"    {name}:")
            print(f"      Hits@{k}: {m.hits_at_k:.3f}")
            print(f"      Precision@{k}: {m.precision_at_k:.3f}")
            print(f"      Recall@{k}: {m.recall_at_k:.3f}")
            print(f"      MRR: {m.mrr:.3f}")
        
        # Save metrics
        save_metrics(metrics, self.config.metrics_dir / "evaluation_metrics.json")
        
        stage_elapsed = time.time() - stage_start
        print(f"  Stage 4 time: {stage_elapsed:.1f}s")
        
        return metrics
    
    def _generate_plots(
        self,
        trigger_results: Dict[str, Any],
        diagnosis_results: Optional[Dict[str, Any]],
    ):
        """
        Generate all standard plots for the experiment.
        
        Args:
            trigger_results: Results from trigger detection
            diagnosis_results: Results from diagnosis (may be None if skipped)
        """
        print("\n  Generating plots...")
        plots_dir = self.config.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        t0 = self.config.drift.t0
        trigger_round = trigger_results.get('trigger_round')
        feature_names = self.trainer.get_feature_names()
        ground_truth = self.config.drift.drifted_features
        if not ground_truth:
            ground_truth = self.trainer.get_drifted_feature_indices()
        
        # 1. Training curves (loss plot)
        try:
            global_loss = self.trainer.global_loss_series
            client_loss = np.load(self.config.logs_dir / "client_loss_matrix.npy")
            plot_training_curves(
                global_loss=global_loss,
                client_loss_matrix=client_loss,
                t0=t0,
                trigger_round=trigger_round,
                save_path=plots_dir / "training_curves.png",
            )
            print(f"    Saved: training_curves.png")
        except Exception as e:
            print(f"    Warning: Could not generate training_curves.png: {e}")
        
        # 2. Loss + RDS detection plot (always generated)
        try:
            loss_series = trigger_results.get('loss_series', [])
            rds_scores = trigger_results.get('rds_scores', [])
            rds_threshold = trigger_results.get('methods', {}).get('rds', {}).get('threshold')
            rds_threshold_series = trigger_results.get('rds_threshold_series', [])
            
            if loss_series:
                plot_loss_and_rds_detection(
                    loss_series=loss_series,
                    rds_scores=rds_scores if rds_scores else [],
                    t0=t0,
                    trigger_round=trigger_round,
                    threshold=rds_threshold,
                    threshold_series=rds_threshold_series if rds_threshold_series else None,
                    calibration_start=self.config.trigger.calibration_start_round,
                    calibration_end=self.config.trigger.calibration_end_round,
                    warmup_rounds=self.config.trigger.warmup_rounds,
                    save_path=plots_dir / "loss_rds_detection.png",
                )
                print(f"    Saved: loss_rds_detection.png")
        except Exception as e:
            print(f"    Warning: Could not generate loss_rds_detection.png: {e}")
        
        # Skip FI plots if no diagnosis
        if diagnosis_results is None:
            print("    Skipping FI plots (no diagnosis results)")
            return
        
        diagnosis_rounds = diagnosis_results.get('diagnosis_rounds', [])
        fi_matrices = diagnosis_results.get('fi_matrices', {})
        
        # 2. FI heatmaps for each method
        for method, fi_matrix in fi_matrices.items():
            try:
                plot_fi_heatmap(
                    fi_matrix=fi_matrix,
                    feature_names=feature_names,
                    method_name=method.upper(),
                    diagnosis_rounds=diagnosis_rounds,
                    save_path=plots_dir / f"fi_{method}.png",
                )
                print(f"    Saved: fi_{method}.png")
            except Exception as e:
                print(f"    Warning: Could not generate fi_{method}.png: {e}")
        
        # 3. Diagnosis rankings plot
        try:
            rankings = {}
            for name, result in diagnosis_results.get('dist_fi', {}).items():
                rankings[name] = result.feature_ranking
            for name, result in diagnosis_results.get('mean_fi_ph', {}).items():
                rankings[name] = result.feature_ranking
            
            if rankings:
                plot_diagnosis_results(
                    rankings=rankings,
                    ground_truth=ground_truth,
                    feature_names=feature_names,
                    save_path=plots_dir / "diagnosis_rankings.png",
                )
                print(f"    Saved: diagnosis_rankings.png")
        except Exception as e:
            print(f"    Warning: Could not generate diagnosis_rankings.png: {e}")
        
        # 4. RDS scores plot
        try:
            rds_scores = {}
            for name, result in diagnosis_results.get('dist_fi', {}).items():
                rds_scores[name] = result.rds_scores
            
            if rds_scores:
                plot_rds_scores(
                    rds_scores=rds_scores,
                    feature_names=feature_names,
                    ground_truth=ground_truth,
                    save_path=plots_dir / "rds_scores.png",
                )
                print(f"    Saved: rds_scores.png")
        except Exception as e:
            print(f"    Warning: Could not generate rds_scores.png: {e}")
        
        # 5. FI + RDS detection plots (similar to loss + RDS detection)
        # One plot per FI method showing FI values and their RDS over diagnosis window
        for method, fi_matrix in fi_matrices.items():
            try:
                dist_fi_key = f'dist_{method}'
                if dist_fi_key in diagnosis_results.get('dist_fi', {}):
                    dist_result = diagnosis_results['dist_fi'][dist_fi_key]
                    
                    # Only plot if we have calibration data
                    if dist_result.rds_series is not None and dist_result.thresholds is not None:
                        plot_fi_and_rds_detection(
                            fi_matrix=fi_matrix,
                            rds_series=dist_result.rds_series,
                            rds_rounds=dist_result.rds_rounds,
                            diagnosis_rounds=diagnosis_rounds,
                            thresholds=dist_result.thresholds,
                            feature_names=feature_names,
                            method_name=method.upper(),
                            ground_truth=ground_truth,
                            trigger_round=trigger_round,
                            calibration_mu=dist_result.calibration_mu,
                            calibration_sigma=dist_result.calibration_sigma,
                            save_path=plots_dir / f"fi_rds_detection_{method}.png",
                        )
                        print(f"    Saved: fi_rds_detection_{method}.png")
            except Exception as e:
                print(f"    Warning: Could not generate fi_rds_detection_{method}.png: {e}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete experiment pipeline.
        
        Returns:
            Complete experiment results
        """
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {self.config.experiment_name}")
        print("=" * 60)
        
        # Stage 1: Training
        training_summary = self.run_training()
        
        # Stage 2: Trigger Detection
        trigger_results = self.run_trigger_detection()
        
        # Stage 2b: FL Baseline Detection
        fl_baseline_results = self.run_fl_baselines()
        
        skip_fi_and_diagnosis = (
            self.config.trigger.skip_diagnosis_if_no_trigger
            and not trigger_results.get('triggered', False)
        )
        diagnosis_time = 0.0
        eval_time = 0.0
        if skip_fi_and_diagnosis:
            print("\n  Skipping FI computation and diagnosis (Stage 1 drift not triggered).")
            diagnosis_results = None
            metrics = None
        else:
            # Stage 3: Diagnosis
            diagnosis_results = self.run_diagnosis()
            diagnosis_time = diagnosis_results.get('stage_time_seconds', 0.0)
            # Stage 4: Evaluation
            metrics = self.run_evaluation()
        
        # Stage 5: Generate Plots
        plot_start = time.time()
        self._generate_plots(trigger_results, diagnosis_results)
        plot_elapsed = time.time() - plot_start
        print(f"  Stage 5 (plots) time: {plot_elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        
        # Timing summary
        training_time = training_summary.get('stage_time_seconds', 0.0)
        trigger_time = trigger_results.get('stage_time_seconds', 0.0)
        fl_baseline_time = fl_baseline_results.get('stage_time_seconds', 0.0)
        
        print("\n" + "=" * 60)
        print(f"EXPERIMENT COMPLETE")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Timing breakdown:")
        print(f"    Stage 1 (Training):      {training_time:.1f}s ({100*training_time/elapsed:.1f}%)")
        print(f"    Stage 2 (Trigger):       {trigger_time:.1f}s ({100*trigger_time/elapsed:.1f}%)")
        print(f"    Stage 2b (FL Baselines): {fl_baseline_time:.1f}s ({100*fl_baseline_time/elapsed:.1f}%)")
        if diagnosis_time > 0:
            print(f"    Stage 3 (Diagnosis):     {diagnosis_time:.1f}s ({100*diagnosis_time/elapsed:.1f}%)")
        print(f"    Stage 5 (Plots):         {plot_elapsed:.1f}s ({100*plot_elapsed/elapsed:.1f}%)")
        print(f"  Results saved to: {self.config.output_dir}")
        print("=" * 60)
        
        # Compile final results
        final_results = {
            'experiment_name': self.config.experiment_name,
            'seed': self.config.seed,
            'elapsed_seconds': elapsed,
            'training': training_summary,
            'trigger': trigger_results,
            'fl_baselines': {
                name: method_result['trigger_metrics']
                for name, method_result in fl_baseline_results.get('methods', {}).items()
            },
            'skip_diagnosis_no_trigger': skip_fi_and_diagnosis,
        }
        if diagnosis_results is not None:
            final_results['diagnosis'] = {
                'trigger_round': diagnosis_results['trigger_round'],
                'diagnosis_rounds': diagnosis_results['diagnosis_rounds'],
            }
        if metrics is not None:
            final_results['metrics'] = {
                name: {
                    'hits_at_k': m.hits_at_k,
                    'precision_at_k': m.precision_at_k,
                    'recall_at_k': m.recall_at_k,
                    'mrr': m.mrr,
                }
                for name, m in metrics.items()
            }
        
        # Save final summary
        with open(self.config.output_dir / "experiment_summary.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        return final_results


class BatchExperimentRunner:
    """
    Runner for batch experiments across multiple configurations.
    """
    
    def __init__(
        self,
        configs: List[ExperimentConfig],
    ):
        self.configs = configs
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all experiments and aggregate results.
        
        Returns:
            Aggregated results
        """
        print(f"\nRunning {len(self.configs)} experiments...")
        
        for i, config in enumerate(self.configs):
            print(f"\n{'#' * 60}")
            print(f"EXPERIMENT {i+1}/{len(self.configs)}")
            print(f"{'#' * 60}")
            
            runner = ExperimentRunner(config)
            result = runner.run()
            self.results.append({
                'config': config,
                'result': result,
                'metrics': runner.metrics_results,
            })
        
        # Aggregate results
        aggregated = self._aggregate_results()
        
        return aggregated
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all runs."""
        # Group by drift configuration
        by_drift_config: Dict[str, List[Dict[str, Any]]] = {}
        
        for r in self.results:
            config = r['config']
            key = f"drift_{config.drift.drifted_client_proportion}_mag_{config.drift.drift_magnitude}"
            
            if key not in by_drift_config:
                by_drift_config[key] = []
            by_drift_config[key].append(r)
        
        # Aggregate metrics for each configuration
        aggregated = {}
        
        for drift_key, runs in by_drift_config.items():
            metrics_list = [r['metrics'] for r in runs if r['metrics']]
            
            if metrics_list:
                agg_metrics = aggregate_results(metrics_list)
                aggregated[drift_key] = agg_metrics
                
                print(f"\n{drift_key}:")
                print(create_results_table(agg_metrics))
        
        # Save aggregated results
        if self.configs:
            base_dir = self.configs[0].base_output_dir
            summary_dir = base_dir / "aggregated"
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            for drift_key, agg_metrics in aggregated.items():
                save_aggregated_metrics(
                    agg_metrics,
                    summary_dir / f"aggregated_{drift_key}.json"
                )
        
        return aggregated


def run_hyperplane_experiments(
    seeds: List[int] = [42, 123, 456, 789, 101112],
    drift_proportions: List[float] = [0.1, 0.2, 0.4, 0.8],
    drift_magnitudes: List[float] = [0.3, 0.5, 0.7],
    drifted_features: Set[int] = {0, 1},  # First two features for Hyperplane
    base_output_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Run hyperplane experiments with specified configurations.
    """
    configs = create_experiment_configs(
        dataset_name='hyperplane',
        seeds=seeds,
        drift_proportions=drift_proportions,
        drift_magnitudes=drift_magnitudes,
        drifted_features=drifted_features,
        base_output_dir=base_output_dir,
    )
    
    runner = BatchExperimentRunner(configs)
    return runner.run_all()


def run_agrawal_experiments(
    seeds: List[int] = [42, 123, 456, 789, 101112],
    drift_proportions: List[float] = [0.1, 0.2, 0.4, 0.8],
    drifted_features: Set[int] = {0, 2, 3},  # salary, age, elevel for function switch 0->2
    base_output_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Run Agrawal experiments with specified configurations.
    """
    configs = create_experiment_configs(
        dataset_name='agrawal',
        seeds=seeds,
        drift_proportions=drift_proportions,
        drift_magnitudes=[1.0],  # Not used for Agrawal
        drifted_features=drifted_features,
        base_output_dir=base_output_dir,
    )
    
    runner = BatchExperimentRunner(configs)
    return runner.run_all()
