"""Fixed-seed adaptation task execution and native-query verification.

Worker library only. Its caller must verify the amended Tier3/protocol/base
inputs, hold the shared GPU lock and assign one authorized device. This module
does not choose epochs, schedule tasks, pool supports or claim study completion.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .run import ROOT, WORK, sha256
from . import adaptation_fit as fit, adaptation_query as query, adaptation_protocol as protocol
from . import control_queue
from .control_plan import GPU_UUIDS
from .checkpoint_evaluation import parameter_digest, read_native_predictions

read, immutable = control_queue.read, control_queue.immutable


def inventory(adaptation, records):
    """Every frozen support fit and unique k=0 cohort, without materializing tokens."""
    if adaptation['seeds'] != [42] or adaptation['epochs'] != 15 or set(records) != {'A', 'B'}:
        raise ValueError('Adaptation requires the amended A/B seed42 protocol')
    jobs, baselines = [], {}
    for cell in adaptation['cells']:
        if (cell['selection_regime'], cell['selection_seed'], cell['selection_role']) != ('B', 42, 'validation'):
            raise ValueError('Only frozen B42 validation selects adaptation epochs')
        if cell['scope'] not in protocol.SCOPES or cell['kind'] not in ('across', 'within'):
            raise ValueError('Unknown adaptation cell')
        counts = {}
        for regime in ('A', 'B'):
            bound = cell['task_bindings'][regime]
            expected_roles = {'across'} if cell['kind'] == 'across' else {'validation', 'test'}
            if set(bound) != expected_roles: raise ValueError('Frozen task binding roles changed')
            ids = [identity for values in bound.values() for identity in values]
            if len(ids) != len(set(ids)): raise ValueError('Duplicate support fit in a cell')
            counts[regime] = len(ids)
            for identity in ids:
                task = records[regime]['tasks'][identity]
                if (task['kind'], task['k']) != (cell['kind'], cell['k']) or not task['support_view']:
                    raise ValueError('Support task differs from its cell')
                roles = ['validation', 'test'] if task['kind'] == 'across' else [task['probe_role']]
                if task['kind'] == 'within' and identity not in bound[roles[0]]:
                    raise ValueError('Within-unit task assigned to a different role')
                selection = regime == 'B' and 'validation' in roles
                job_id = f"{regime}__{cell['id']}__{identity}"
                if Path(job_id).name != job_id: raise ValueError('Task identifier escapes its directory')
                job = dict(id=job_id, regime=regime, seed=42, cell_id=cell['id'], task_id=identity,
                    kind=cell['kind'], scope=cell['scope'], k=cell['k'],
                    selection_fit=selection, query_roles=roles, baseline_task=task['baseline_task'])
                jobs.append(job)
                baseline = records[regime]['tasks'][task['baseline_task']]
                if baseline['kind'] != task['kind'] or baseline['k'] != 0 or baseline['support_view'] is not None:
                    raise ValueError('Adaptation baseline must be unchanged k=0')
                for role in roles:
                    view_key = 'validation_view' if task['kind'] == 'across' and role == 'validation' else 'query_view'
                    if task[view_key] != baseline[view_key]: raise ValueError('Baseline uses a different query cohort')
                    key = f"{regime}__{task['baseline_task']}__{role}"
                    value = dict(id=key, regime=regime, seed=42, task_id=task['baseline_task'], role=role,
                        query_view=task[view_key], selection_phase=regime == 'B' and role == 'validation')
                    if key in baselines and baselines[key] != value: raise ValueError('Conflicting baseline requests')
                    baselines[key] = value
        if counts != cell['fits_per_seed'] or sum(counts.values()) != cell['total_fits']:
            raise ValueError('Requested adaptation fit counts differ from frozen cells')
    if (len(jobs) != adaptation['accounting']['support_fits'] or
        len({j['id'] for j in jobs}) != len(jobs)):
        raise ValueError('Requested adaptation fit inventory is incomplete or duplicated')
    selection = sum(j['selection_fit'] for j in jobs)
    remaining_queries = sum(len(j['query_roles']) - int(j['selection_fit']) for j in jobs)
    return dict(fits=jobs, baselines=sorted(baselines.values(), key=lambda x: x['id']),
        accounting=dict(support_fits=len(jobs), selection_fits=selection, remaining_fits=len(jobs)-selection,
            selection_queries=15*selection, remaining_selected_queries=remaining_queries,
            baseline_queries=len(baselines), maximum_adapter_snapshots=15*selection+2*(len(jobs)-selection)),
        unavailable=adaptation['unavailable'],
        interpretation='Counts preserve all frozen tasks, including undefined query cohorts. Later fits/selected queries depend on a selectable B validation epoch; no test-based fallback. k=0 is shared across scopes and budgets only on the identical query view.')


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class BaseState:
    """Restore only allowed adaptation parameters; frozen-state drift is fatal."""
    def __init__(self, model):
        self.model = model
        self.state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        self.digest = tensor_digest(self.state)
        if parameter_digest(model) != self.digest: raise ValueError('Base-state copy differs from loaded model')
        self.names = {s: fit.scope_parameters(model, s) for s in protocol.SCOPES}
        self.frozen = {s: tensor_digest({n: t for n, t in self.state.items() if n not in names})
            for s, names in self.names.items()}
        self.snapshot_digests = {}

    def restore(self):
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name in self.names['heads_cross_attention']:
                parameters[name].copy_(self.state[name])
                parameters[name].grad = None
        if parameter_digest(self.model) != self.digest:
            raise RuntimeError('Adaptation changed frozen base state; refusing the next task')

    def adapter_digest(self, path, checksum, *, scope, epoch, step):
        """Hash the exact base+adapter state without copying encoders across PCIe."""
        path = Path(path)
        if sha256(path) != checksum: raise ValueError('Adapter payload changed')
        key = checksum, scope, epoch, step
        if key in self.snapshot_digests: return self.snapshot_digests[key]
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['format_version'] != 1 or payload['scope'] != scope or payload['epoch'] != epoch or
            payload['step'] != step or payload['base_parameter_sha256'] != self.digest or
            payload['frozen_parameter_sha256'] != self.frozen[scope] or
            sorted(payload['parameters']) != self.names[scope]):
            raise ValueError('Adapter does not belong to this base, scope or epoch')
        for name, tensor in payload['parameters'].items():
            if (tensor.shape != self.state[name].shape or tensor.dtype != self.state[name].dtype or
                not torch.isfinite(tensor).all()): raise ValueError('Invalid adapter tensor')
        value = tensor_digest(dict(self.state, **payload['parameters']))
        self.snapshot_digests[key] = value
        return value


class TaskRuntime:
    """One verified loaded base and view bundle, reused for independent tasks.

    Production callers own the Tier3, checkpoint and scheduler gates. The
    explicit source record binds those gates into each operation's receipt.
    """
    def __init__(self, model, views, config, output, source, *, cpu=False):
        if not isinstance(views, query.AdaptationViews) or config.training.seed != 42:
            raise ValueError('Use verified adaptation views and seed42')
        if config.model != model.config or config.model.contrastive_mask_policy == 'training_known':
            raise ValueError('Use the resolved full-reference model configuration without a new masking policy')
        if not source or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Task runtime needs explicit provenance and an independent worker')
        output = Path(output)
        if cpu:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(p.device.type != 'cpu' for p in model.parameters()):
                raise ValueError('CPU qualification requires a CPU model and hidden GPUs')
        elif (ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') not in GPU_UUIDS or
            not output.absolute().is_relative_to(WORK) or any(p.device.type != 'cuda' for p in model.parameters()) or
            config.training.num_train_epochs != 15 or config.training.per_device_train_batch_size != 12 or
            config.training.gradient_accumulation_steps != 16):
            raise ValueError('Scientific adaptation requires the fixed15-epoch 12x16 protocol on one authorized Atum GPU')
        self.model, self.views, self.config, self.output = model, views, config, output
        self.source, self.cpu, self.base = source, cpu, BaseState(model)

    def check_job(self, job):
        task = self.views.record['tasks'][job['task_id']]
        if (job['regime'] != self.views.record['regime'] or job['seed'] != 42 or
            (job['kind'], job['k']) != (task['kind'], task['k']) or job['scope'] not in protocol.SCOPES or
            job['baseline_task'] != task['baseline_task'] or
            job['query_roles'] != (['validation', 'test'] if task['kind'] == 'across' else [task['probe_role']]) or
            job['selection_fit'] != (job['regime'] == 'B' and 'validation' in job['query_roles']) or
            Path(job['id']).name != job['id']):
            raise ValueError('Task execution differs from its frozen support/query binding')
        return task

    def fit_directory(self, job):
        return self.output / 'fits' / job['id'] / 'native'

    def verify_fit(self, job, save_epochs, *, require_receipt=True):
        task = self.check_job(job); directory = self.fit_directory(job)
        done = read(directory / 'complete.json'); plan = read(directory / 'fit_plan.json')
        source = query.task_source(self.views, job['task_id'], 42)
        training = replace(self.config.training, output_dir=str(directory), training_mode='single_gpu',
            fixed_train_eval_assays=0, export_evaluation_predictions=False,
            objective_gradient_diagnostics_steps=0, evaluate_test=False)
        if self.cpu: training = replace(training, bf16=False, fp16=False, dataloader_num_workers=0, report_to=[])
        if (done['fit_plan_sha256'] != sha256(directory / 'fit_plan.json') or
            done['source'] != source or plan['source'] != source or done['scope'] != job['scope'] or plan['scope'] != job['scope'] or
            done['base_parameter_sha256'] != self.base.digest or plan['base_parameter_sha256'] != self.base.digest or
            plan['frozen_parameter_sha256'] != self.base.frozen[job['scope']] or
            not done['frozen_parameters_unchanged'] or done['query_examples_received'] or done['selection_performed'] or
            plan['query_examples_received'] or plan['selection_performed'] or
            done['cpu_qualification'] != self.cpu or plan['cpu_qualification'] != self.cpu or
            plan['training'] != json.loads(json.dumps(asdict(training))) or
            plan['data'] != json.loads(json.dumps(asdict(self.config.data))) or
            plan['model_config'] != json.loads(json.dumps(self.model.config.to_dict())) or
            plan['worker_source_sha256'] != sha256(Path(fit.__file__)) or
            plan['support_fingerprint'] != source['support_fingerprint'] or
            plan['save_epochs'] != list(save_epochs) or [s['epoch'] for s in done['snapshots']] != list(save_epochs) or
            done['epochs'] != training.num_train_epochs or done['trainable_parameters'] != self.base.names[job['scope']] or
            plan['trainable_parameters'] != self.base.names[job['scope']] or
            done['support_assays'] != len(self.views.record['views'][task['support_view']]['group_ids'])):
            raise ValueError('Support fit provenance, scope, horizon or snapshot inventory changed')
        for snapshot in done['snapshots']:
            path = directory / snapshot['file']
            if path.resolve().parent != directory.resolve() or path.stat().st_size != snapshot['bytes']:
                raise ValueError('Support snapshot location or size changed')
            if sha256(path) != snapshot['sha256']: raise ValueError('Support snapshot changed')
        if require_receipt:
            receipt = read(directory.parent / 'receipt.json')
            if (receipt['source'] != self.source or receipt['job'] != job or
                receipt['native_complete_sha256'] != sha256(directory / 'complete.json') or
                receipt['base_parameter_sha256'] != self.base.digest or not receipt['base_restored'] or
                receipt['cpu_qualification'] != self.cpu):
                raise ValueError('Support fit belongs to another execution or was not restored')
        return done

    def fit_task(self, job, save_epochs):
        task = self.check_job(job); directory = self.fit_directory(job)
        expected = list(range(1, int(self.config.training.num_train_epochs)+1)) if job['selection_fit'] else list(save_epochs)
        if list(save_epochs) != expected:
            raise ValueError('B validation selection needs every positive adaptation epoch')
        self.base.restore()
        if (directory / 'complete.json').exists():
            self.verify_fit(job, save_epochs)
            receipt = read(directory.parent / 'receipt.json')
            if receipt['source'] != self.source or receipt['job'] != job or receipt['native_complete_sha256'] != sha256(directory / 'complete.json'):
                raise ValueError('Support fit belongs to another execution')
            return receipt
        if directory.parent.exists(): raise RuntimeError('Partial adaptation fit requires inspection; no implicit retry')
        try:
            fit.fit_support(self.model, self.views.load(task['support_view']), self.config, directory,
                scope=job['scope'], base_parameter_sha256=self.base.digest,
                source=query.task_source(self.views, job['task_id'], 42), save_epochs=list(save_epochs), cpu=self.cpu)
        finally:
            self.base.restore()
        done = self.verify_fit(job, save_epochs, require_receipt=False)
        receipt = dict(source=self.source, job=job, native_complete_sha256=sha256(directory / 'complete.json'),
            base_parameter_sha256=self.base.digest, base_restored=True, cpu_qualification=self.cpu,
            native_epochs=done['epochs'], native_steps=done['global_step'], query_scoring_complete=False)
        immutable(directory.parent / 'receipt.json', receipt)
        return receipt

    def query_directory(self, task_id, role, epoch, scope):
        if scope is None: return self.output / 'baselines' / self.views.record['regime'] / task_id / role
        return self.output / 'queries' / self.views.record['regime'] / task_id / scope / role / f'epoch{epoch:02d}'

    def score(self, task_id, role, epoch, *, job=None):
        task = self.views.record['tasks'][task_id]; baseline = task['k'] == 0
        if baseline:
            if job is not None or epoch != 0: raise ValueError('k=0 never fits or selects an epoch')
            scope = None; directory = None; checksum = None
        else:
            if job is None or job['task_id'] != task_id: raise ValueError('Adapted query needs its exact support job')
            self.check_job(job)
            if role not in job['query_roles']: raise ValueError('Query role differs from the support task')
            scope = job['scope']; directory = self.fit_directory(job); checksum = sha256(directory / 'complete.json')
        output = self.query_directory(task_id, role, epoch, scope)
        self.base.restore()
        if not (output / 'complete.json').exists():
            if output.exists(): raise RuntimeError('Partial adaptation query requires inspection; no implicit retry')
            query.evaluate_query(self.model, self.views, self.config, output,
                task_id=task_id, role=role, epoch=epoch, scope=scope,
                base_parameter_sha256=self.base.digest, fit_directory=directory,
                fit_complete_sha256=checksum, cpu=self.cpu)
        result = self.verify_query(task_id, role, epoch, job=job)
        immutable(output / 'execution.json', dict(source=self.source,
            query_complete_sha256=sha256(output / 'complete.json'), base_parameter_sha256=self.base.digest,
            cpu_qualification=self.cpu, raw_predictions_and_adapter_verified=True))
        return result

    def verify_query(self, task_id, role, epoch, *, job=None):
        task = self.views.record['tasks'][task_id]; baseline = task['k'] == 0
        scope = None if baseline else job['scope']
        output = self.query_directory(task_id, role, epoch, scope)
        result = read(output / 'complete.json'); native = read(output / 'native/complete.json')
        name, examples, groups = query.query_view(self.views, task_id, role)
        source = dict(query.task_source(self.views, task_id, 42), query_view=name, query_role=role,
            query_fingerprint=examples._fingerprint, epoch=epoch, scope=scope)
        expected_digest, adapter_hash, fit_hash = self.base.digest, None, None
        if not baseline:
            self.check_job(job); directory = self.fit_directory(job); fit_hash = sha256(directory / 'complete.json')
            fit_done = read(directory / 'complete.json')
            self.verify_fit(job, read(directory / 'fit_plan.json')['save_epochs'])
            matches = [s for s in fit_done['snapshots'] if s['epoch'] == epoch]
            if len(matches) != 1: raise ValueError('Selected query snapshot is missing or duplicated')
            snapshot = matches[0]; adapter_hash = snapshot['sha256']
            expected_digest = self.base.adapter_digest(directory / snapshot['file'], adapter_hash,
                scope=scope, epoch=epoch, step=snapshot['step'])
        if (result['source'] != source or native['source'] != source or
            result['worker_source_sha256'] != sha256(Path(query.__file__)) or
            result['base_parameter_sha256'] != self.base.digest or native['parameter_sha256'] != expected_digest or
            result['fit_complete_sha256'] != fit_hash or result['adapter_sha256'] != adapter_hash or
            not result['base_restored'] or not result['modes_and_rng_restored'] or not native['parameters_unchanged'] or
            result['optimizer_created'] or result['selection_performed'] or native['optimizer_created'] or
            result['cpu_qualification'] != self.cpu or native['cpu_qualification'] != self.cpu or
            native['world_size'] != 1 or not native['device'].startswith('cpu' if self.cpu else 'cuda') or
            native['ranking_partition_seed'] != 42 or native['ranking_num_partitions'] != self.config.data.evaluation_ranking_partitions or
            native['contrastive_assay_batch_size'] != self.config.training.per_device_train_batch_size or
            native['bf16'] != (False if self.cpu else self.config.training.bf16) or
            native['fp16'] != (False if self.cpu else self.config.training.fp16) or
            result['native_complete_sha256'] != sha256(output / 'native/complete.json')):
            raise ValueError('Query provenance, parameter state or native execution changed')
        manifest = Path(result['prediction_manifest'])
        prefix = 'eval' if role == 'validation' else 'test'
        expected_manifest = output / 'native/prediction_exports/fixed_checkpoint' / f'{prefix}_step00000000.json'
        if (manifest != expected_manifest or native['prediction_manifest'] != str(manifest) or
            result['prediction_manifest_sha256'] != sha256(manifest) or native['prediction_manifest_sha256'] != sha256(manifest)):
            raise ValueError('Native query manifest changed')
        frame, metadata = read_native_predictions(manifest)
        if (metadata['dataset_fingerprint'] != examples._fingerprint or metadata['metric_prefix'] != prefix or
            native['prediction_payload_sha256'] != metadata['predictions_sha256']):
            raise ValueError('Native query payload no longer belongs to its view')
        expected = query.example_frame(examples.select_columns(query.IDENTITY + query.LABELS).to_pandas())
        actual = query.example_frame(frame)
        pd.testing.assert_frame_equal(actual[query.IDENTITY + query.LABELS], expected[query.IDENTITY + query.LABELS],
            check_dtype=False, check_exact=True)
        pathways, records = query.pathway_records(actual, groups, self.config.data.ranking_min_pchembl_span)
        if pathways != result['pathways'] or records != result['assay_records']:
            raise ValueError('Saved query metrics differ from raw native predictions')
        # Independent rank calculation validates the reported within-assay values.
        by_group = dict(tuple(actual.groupby('group_id', sort=False)))
        for record in records:
            rows = by_group[record['group_id']]; values = rows[record['scorer']].to_numpy()
            labels = rows.pchembl_value.to_numpy(dtype=np.float32).astype(np.float64)
            rho = float(spearmanr(values, labels).statistic) if record['eligible'] and np.ptp(values) > 0 else None
            rho = rho if rho is not None and math.isfinite(rho) else None
            if (rho is None) != (record['rho'] is None) or (rho is not None and not math.isclose(rho, record['rho'], rel_tol=0, abs_tol=1e-14)):
                raise ValueError('Native assay correlation disagrees with independent SciPy ranks')
        selection = [dict(regime=source['regime'], seed=42, role=role, kind=task['kind'], scope=scope,
            k=task['k'], epoch=epoch, group_id=r['group_id'], unit=r['unit'], rho=r['rho'])
            for r in records if r['scorer'] == 'ranking_score' and r['eligible']] if not baseline else []
        if selection != result['selection_records'] or (result['rows'], result['assays'], result['units']) != (len(examples), len(groups), int(groups.unit.nunique())):
            raise ValueError('Selection records or complete query coverage changed')
        return result
