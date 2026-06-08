import math
import time

from transformers import Trainer
import torch
import torch.nn.functional as F


class GPT2_w_crs_attn_Trainer(Trainer):
    """
    Custom trainer for the Prot2Mol model.
    
    This trainer works with the unified Prot2MolModel that combines
    protein encoder and molecule decoder.
    """
    
    def __init__(
        self,
        *args,
        training_stage="multitask",
        compute_generation_metrics=None,
        ignore_mismatched_optimizer=False,
        pchembl_pair_weight=1.0,
        **kwargs,
    ):
        """
        Initialize the trainer with explicit objective-stage support.
        
        Args:
            training_stage: Objective family used for this run.
            ignore_mismatched_optimizer: If True, skip loading optimizer state when it doesn't match
        """
        super().__init__(*args, **kwargs)
        self.training_stage = training_stage
        self.compute_generation_metrics = compute_generation_metrics
        self.ignore_mismatched_optimizer = ignore_mismatched_optimizer
        self.pchembl_pair_weight = pchembl_pair_weight
        # Storage for pChEMBL predictions during evaluation
        self.pchembl_predictions_list = []
        self.pchembl_targets_list = []
        self.pchembl_group_ids_list = []
        self._reset_train_component_accumulator()
        self._reset_eval_component_accumulator()

    @staticmethod
    def _is_same_loss_value(primary_loss, component_loss):
        if primary_loss is None or component_loss is None:
            return False
        try:
            return math.isclose(float(primary_loss), float(component_loss), rel_tol=1e-6, abs_tol=1e-8)
        except (TypeError, ValueError):
            return False

    def _stage_has_lm(self):
        return self.training_stage in {"lm_only", "multitask"}

    def _stage_has_pchembl(self):
        return self.training_stage in {"pchembl_only", "multitask"}

    def _reset_train_component_accumulator(self):
        self._train_component_sums = {
            "lm_loss": 0.0,
            "pchembl_loss": 0.0,
            "pair_loss": 0.0,
            "total_loss": 0.0,
        }
        self._train_component_count = 0

    def _reset_eval_component_accumulator(self):
        self._eval_component_sums = {
            "lm_loss": 0.0,
            "pchembl_loss": 0.0,
            "pair_loss": 0.0,
            "total_loss": 0.0,
        }
        self._eval_component_count = 0

    @staticmethod
    def _to_log_scalar(value):
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu().item())
        return float(value)

    def _record_train_components(self, lm_loss, pchembl_loss, pair_loss, total_loss):
        self._train_component_sums["lm_loss"] += self._to_log_scalar(lm_loss)
        self._train_component_sums["pchembl_loss"] += self._to_log_scalar(pchembl_loss)
        self._train_component_sums["pair_loss"] += self._to_log_scalar(pair_loss)
        self._train_component_sums["total_loss"] += self._to_log_scalar(total_loss)
        self._train_component_count += 1

    def _record_eval_components(self, lm_loss, pchembl_loss, pair_loss, total_loss):
        self._eval_component_sums["lm_loss"] += self._to_log_scalar(lm_loss)
        self._eval_component_sums["pchembl_loss"] += self._to_log_scalar(pchembl_loss)
        self._eval_component_sums["pair_loss"] += self._to_log_scalar(pair_loss)
        self._eval_component_sums["total_loss"] += self._to_log_scalar(total_loss)
        self._eval_component_count += 1

    def _consume_train_component_logs(self, primary_loss=None):
        if self._train_component_count == 0:
            return {}
        denom = float(self._train_component_count)
        logs = {
            "lm_loss": self._train_component_sums["lm_loss"] / denom,
            "pchembl_loss": self._train_component_sums["pchembl_loss"] / denom,
            "ranking_loss": self._train_component_sums["pair_loss"] / denom,
            "total_loss": self._train_component_sums["total_loss"] / denom,
        }
        if self._is_same_loss_value(primary_loss, logs["total_loss"]):
            logs.pop("total_loss")
        self._reset_train_component_accumulator()
        return logs

    def _consume_eval_component_logs(self, metric_key_prefix, primary_loss=None):
        is_distributed = getattr(self.args, "local_rank", -1) != -1
        if self._eval_component_count == 0 and not is_distributed:
            return {}

        sums = torch.tensor(
            [
                self._eval_component_sums["lm_loss"],
                self._eval_component_sums["pchembl_loss"],
                self._eval_component_sums["pair_loss"],
                self._eval_component_sums["total_loss"],
                float(self._eval_component_count),
            ],
            device=self.args.device,
            dtype=torch.float64,
        )

        if is_distributed:
            import torch.distributed as dist

            dist.all_reduce(sums)

        if sums[-1].item() <= 0:
            self._reset_eval_component_accumulator()
            return {}

        denom = float(sums[-1].item())
        logs = {
            f"{metric_key_prefix}_lm_loss": float(sums[0].item() / denom),
            f"{metric_key_prefix}_pchembl_loss": float(sums[1].item() / denom),
            f"{metric_key_prefix}_ranking_loss": float(sums[2].item() / denom),
            f"{metric_key_prefix}_total_loss": float(sums[3].item() / denom),
        }
        if self._is_same_loss_value(primary_loss, logs[f"{metric_key_prefix}_total_loss"]):
            logs.pop(f"{metric_key_prefix}_total_loss")
        if self._stage_has_lm():
            try:
                logs[f"{metric_key_prefix}_perplexity"] = float(math.exp(logs[f"{metric_key_prefix}_lm_loss"]))
            except OverflowError:
                logs[f"{metric_key_prefix}_perplexity"] = float("inf")

        self._reset_eval_component_accumulator()
        return logs
        
    def clear_pchembl_predictions(self):
        """Clear accumulated pChEMBL predictions. Called at the start of each evaluation."""
        self.pchembl_predictions_list = []
        self.pchembl_targets_list = []
        self.pchembl_group_ids_list = []
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to clear pChEMBL predictions before evaluation starts.
        This prevents accumulation of predictions from previous evaluations.
        """
        # Clear any accumulated predictions from previous evaluations
        self.clear_pchembl_predictions()
        self._reset_eval_component_accumulator()

        eval_start = time.perf_counter()
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        component_metrics = self._consume_eval_component_logs(
            metric_key_prefix,
            primary_loss=metrics.get(f"{metric_key_prefix}_loss"),
        )
        metrics.update(component_metrics)
        if component_metrics:
            self.log(component_metrics)

        if self.compute_generation_metrics is not None and self._stage_has_lm():
            generation_start = time.perf_counter()
            generation_metrics = self.compute_generation_metrics(eval_dataset or self.eval_dataset)
            prefixed_generation_metrics = {
                f"{metric_key_prefix}_{key}": value for key, value in generation_metrics.items()
            }
            prefixed_generation_metrics[f"{metric_key_prefix}_generation_metrics_time_sec"] = (
                time.perf_counter() - generation_start
            )
            metrics.update(prefixed_generation_metrics)
            self.log(prefixed_generation_metrics)

        metrics[f"{metric_key_prefix}_evaluation_time_sec"] = time.perf_counter() - eval_start
        self.log({f"{metric_key_prefix}_evaluation_time_sec": metrics[f"{metric_key_prefix}_evaluation_time_sec"]})
        return metrics

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if "loss" in logs:
            logs.update(self._consume_train_component_logs(primary_loss=logs["loss"]))
        return super().log(logs, start_time=start_time)

    def _build_model_inputs(self, inputs):
        """Create normalized model inputs shared by train/eval/prediction paths."""
        train_lm = inputs.get("train_lm", None)
        if train_lm is None:
            batch_size = inputs["mol_input_ids"].shape[0]
            train_lm = torch.ones(batch_size, dtype=torch.bool, device=inputs["mol_input_ids"].device)

        return {
            "mol_input_ids": inputs["mol_input_ids"],
            "prot_input_ids": inputs["prot_input_ids"],
            "prot_attention_mask": inputs["prot_attention_mask"],
            "labels": inputs.get("labels", inputs["mol_input_ids"]),
            "pchembl_values": inputs.get("pchembl_values", None),
            "train_lm": train_lm,
            "pchembl_only_mode": self.training_stage == "pchembl_only",
            "compute_pchembl": self._stage_has_pchembl(),
        }

    def _compose_total_loss(self, model, outputs, inputs):
        """
        Compose the final multi-task loss in one place.
        Keeping this logic centralized avoids model/trainer duplication.
        """
        lm_loss = outputs.get("lm_loss", None)
        pchembl_preds = outputs.get("pchembl_predictions", None)
        pchembl_values = inputs.get("pchembl_values", None)
        group_ids = inputs.get("group_id", None)

        pchembl_loss = None
        pair_loss = None
        if pchembl_preds is not None and pchembl_values is not None:
            delta = model._config.get("pchembl_huber_delta", 1.0)
            pchembl_loss = F.smooth_l1_loss(pchembl_preds, pchembl_values, beta=delta)
            if group_ids is not None:
                pair_loss = self._pairwise_huber(pchembl_preds, pchembl_values, group_ids, delta=delta)

        loss = None
        if self._stage_has_lm() and lm_loss is not None:
            loss = model.lm_weight * lm_loss
        if self._stage_has_pchembl() and pchembl_loss is not None:
            total_pair = pair_loss if pair_loss is not None else 0.0
            pchembl_term = model.pchembl_weight * (pchembl_loss + self.pchembl_pair_weight * total_pair)
            loss = pchembl_term if loss is None else loss + pchembl_term

        if loss is None:
            # Keep backward compatibility if model provides a direct scalar loss.
            loss = outputs.get("loss", None)

        return loss, lm_loss, pchembl_loss, pair_loss
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss using the unified model with auxiliary pChEMBL prediction.
        
        Important:
        - LM loss is computed only for positive samples (train_lm=True)
        - pChEMBL loss is computed for ALL samples (both positive and negative)
        
        Args:
            model: The Prot2MolModel instance
            inputs: Dictionary containing input data
            return_outputs: Whether to return outputs along with loss
            
        Returns:
            Loss tensor, optionally with model outputs
        """
        model_inputs = self._build_model_inputs(inputs)
        outputs = model(**model_inputs)
        loss, lm_loss, pchembl_loss, pair_loss = self._compose_total_loss(model, outputs, inputs)

        if loss is None:
            raise RuntimeError(
                "Loss is None. Check whether labels/pchembl_values are present for the selected training mode."
            )

        if getattr(model, "training", False):
            self._record_train_components(lm_loss, pchembl_loss, pair_loss, loss)

        outputs["lm_loss"] = lm_loss
        if pchembl_loss is not None:
            outputs["pchembl_loss"] = pchembl_loss
        if pair_loss is not None:
            outputs["pchembl_pair_loss"] = pair_loss
        outputs["loss"] = loss
        outputs["inputs"] = inputs
        return (loss, outputs) if return_outputs else loss

    def _pairwise_huber(self, y_pred, y_true, group_ids, delta=1.0):
        """Compute pairwise Huber loss within each group_id."""
        if group_ids is None:
            return torch.tensor(0.0, device=y_pred.device)

        # Filter invalid group ids
        valid_mask = group_ids >= 0
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=y_pred.device)

        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        group_ids = group_ids[valid_mask]

        losses = []
        for g in torch.unique(group_ids):
            idx = (group_ids == g).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            perm = idx[torch.randperm(idx.numel(), device=idx.device)]
            half = perm.numel() // 2
            if half == 0:
                continue
            a = perm[:half]
            b = perm[half:half * 2]
            dy_pred = y_pred[a] - y_pred[b]
            dy_true = y_true[a] - y_true[b]
            losses.append(F.smooth_l1_loss(dy_pred, dy_true, beta=delta))

        if len(losses) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return torch.stack(losses).mean()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step with auxiliary pChEMBL prediction.
        
        Override to handle model outputs that contain non-tensor values (dicts, booleans, etc.)
        which cannot be padded across processes in DDP.
        
        Important: pChEMBL predictions are stored for ALL samples, regardless of whether
        they are positive (train_lm=True) or negative (train_lm=False).
        
        Args:
            model: The Prot2MolModel instance
            inputs: Dictionary containing input data
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in the inputs
            
        Returns:
            Tuple of (loss, logits, labels) - all tensors or None
        """
        # Configure model for prediction
        model_config = model.config
        original_bos_token_id = model_config.bos_token_id
        original_pad_token_id = model_config.pad_token_id
        original_forced_bos = getattr(model_config, "forced_bos_token_id", None)
        original_forced_eos = getattr(model_config, "forced_eos_token_id", None)
        model.config.bos_token_id = 1
        model.config.pad_token_id = 1
        model.config.forced_bos_token_id = 1
        model.config.forced_eos_token_id = 2
        
        try:
            has_labels = "labels" in inputs
            
            model_inputs = self._build_model_inputs(inputs)
            
            # Forward pass through model
            with torch.no_grad():
                outputs = model(**model_inputs)
            
            # Extract only tensor outputs to avoid padding issues
            # The model returns a dict, but we need loss, logits, and pchembl predictions
            if isinstance(outputs, dict):
                loss, lm_loss, pchembl_loss, pair_loss = self._compose_total_loss(model, outputs, inputs)
                logits = outputs.get("logits", None)
                pchembl_preds = outputs.get("pchembl_predictions", None)
                self._record_eval_components(lm_loss, pchembl_loss, pair_loss, loss)
                
                # Store pChEMBL predictions for ALL samples (positive and negative)
                # We do this here because we can't return them from prediction_step (DDP padding issues)
                # Note: We store predictions regardless of train_lm flag - we want metrics for all samples
                if pchembl_preds is not None and model_inputs.get("pchembl_values") is not None:
                    self.pchembl_predictions_list.append(pchembl_preds.detach().cpu())
                    self.pchembl_targets_list.append(model_inputs["pchembl_values"].detach().cpu())
                    if "group_id" in inputs:
                        self.pchembl_group_ids_list.append(inputs["group_id"].detach().cpu())
            else:
                # Handle tuple outputs
                loss = outputs[0] if len(outputs) > 0 else None
                logits = outputs[1] if len(outputs) > 1 else None
                pchembl_preds = None
            
            # Prepare labels for return
            if has_labels:
                labels = inputs["labels"]
            else:
                labels = None
            
            # If only loss is needed, return just that
            if prediction_loss_only:
                return (loss, None, None)
            
            # Return tensors only (no dicts, booleans, or other non-tensor types)
            # Note: We can't return pchembl_preds here because it would be padded across processes
            # causing the same error. We'll need to handle pChEMBL metrics differently.
            return (loss, logits, labels)
            
        finally:
            # Restore original bos_token_id
            model.config.bos_token_id = original_bos_token_id
            model.config.pad_token_id = original_pad_token_id
            if hasattr(model.config, "forced_bos_token_id"):
                model.config.forced_bos_token_id = original_forced_bos
            if hasattr(model.config, "forced_eos_token_id"):
                model.config.forced_eos_token_id = original_forced_eos
    
    def get_pchembl_predictions(self, gather_across_ranks=True):
        """
        Retrieve accumulated pChEMBL predictions and targets.
        
        Args:
            gather_across_ranks: If True and in DDP mode, gather predictions from all ranks
                               to match what Transformers does with LM predictions
        
        Returns:
            Tuple of (predictions, targets, group_ids) as numpy arrays, or (None, None, None) if empty
        """
        local_rank = getattr(self.args, "local_rank", -1)
        is_distributed = gather_across_ranks and local_rank != -1
        if len(self.pchembl_predictions_list) == 0 and not is_distributed:
            return None, None, None

        # Concatenate all batches on this rank. In DDP, ranks with zero local
        # samples must still enter collectives with empty tensors.
        if len(self.pchembl_predictions_list) > 0:
            predictions = torch.cat(self.pchembl_predictions_list, dim=0)
            targets = torch.cat(self.pchembl_targets_list, dim=0)
        else:
            predictions = torch.empty(0, dtype=torch.float32)
            targets = torch.empty(0, dtype=torch.float32)
        group_ids = None
        if len(self.pchembl_group_ids_list) > 0:
            group_ids = torch.cat(self.pchembl_group_ids_list, dim=0)
        
        local_samples = predictions.shape[0]
        
        # Gather across ranks if requested and in DDP mode
        # NOTE: Transformers gathers LM predictions before calling compute_metrics,
        # so we need to do the same for pChEMBL predictions to ensure counts match
        if is_distributed:
            import torch.distributed as dist
            
            # Move to the trainer device for gathering.
            device = torch.device(getattr(self.args, "device", f"cuda:{local_rank}"))
            predictions = predictions.to(device)
            targets = targets.to(device)
            if group_ids is not None:
                group_ids = group_ids.to(device)
            
            world_size = dist.get_world_size()
            
            # Gather sizes first
            local_size = torch.tensor([predictions.shape[0]], device=device)
            size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            total_size = sum(int(size.item()) for size in size_list)
            if total_size == 0:
                return None, None, None

            has_group_ids = torch.tensor([1 if group_ids is not None else 0], device=device)
            dist.all_reduce(has_group_ids, op=dist.ReduceOp.MAX)
            if bool(has_group_ids.item()) and group_ids is None:
                group_ids = torch.full(
                    (predictions.shape[0],),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
            
            max_size = max([s.item() for s in size_list])
            
            # Pad if needed
            if predictions.shape[0] < max_size:
                padding = torch.zeros(max_size - predictions.shape[0], device=predictions.device)
                predictions = torch.cat([predictions, padding], dim=0)
                targets = torch.cat([targets, padding], dim=0)
                if group_ids is not None:
                    gid_padding = torch.full((max_size - group_ids.shape[0],), -1, device=group_ids.device, dtype=group_ids.dtype)
                    group_ids = torch.cat([group_ids, gid_padding], dim=0)
            
            # Gather from all ranks
            gathered_preds = [torch.zeros_like(predictions) for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(targets) for _ in range(world_size)]
            gathered_groups = [torch.zeros_like(group_ids) for _ in range(world_size)] if group_ids is not None else None
            
            dist.all_gather(gathered_preds, predictions)
            dist.all_gather(gathered_targets, targets)
            if group_ids is not None:
                dist.all_gather(gathered_groups, group_ids)
            
            # Concatenate and trim
            predictions_list = []
            targets_list = []
            groups_list = []
            for i, size in enumerate(size_list):
                predictions_list.append(gathered_preds[i][:size.item()])
                targets_list.append(gathered_targets[i][:size.item()])
                if group_ids is not None:
                    groups_list.append(gathered_groups[i][:size.item()])
            
            predictions = torch.cat(predictions_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            if group_ids is not None:
                group_ids = torch.cat(groups_list, dim=0)
           
        else:
            if local_rank != -1:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Rank {local_rank}: Returning {local_samples} LOCAL pChEMBL predictions (no gathering)")
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        group_ids_np = group_ids.cpu().numpy() if group_ids is not None else None
        
        return predictions_np, targets_np, group_ids_np
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """
        Override to handle mismatched optimizer states gracefully.
        
        When model architecture changes (e.g., different number of layers),
        the optimizer state won't match. This method catches that error and
        starts with a fresh optimizer instead.
        """
        if not self.ignore_mismatched_optimizer:
            # Use default behavior
            return super()._load_optimizer_and_scheduler(checkpoint)
        
        try:
            # Try to load optimizer state
            super()._load_optimizer_and_scheduler(checkpoint)
        except (ValueError, RuntimeError) as e:
            if "doesn't match the size" in str(e) or "size mismatch" in str(e):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Optimizer state doesn't match current model architecture. "
                    f"Starting with fresh optimizer. Error: {e}"
                )
                # Don't load optimizer/scheduler state - will use freshly initialized ones
                # The parent class already initialized them in __init__
            else:
                # Re-raise if it's a different error
                raise
