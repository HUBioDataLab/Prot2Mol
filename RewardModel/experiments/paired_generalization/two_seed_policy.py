"""User-authorized seed42/43 amendment; historical registrations stay immutable."""
from __future__ import annotations

from pathlib import Path

if __package__:
    from .run import ROOT, WORK, sha256
    from . import control_queue, control_plan
else:
    from run import ROOT, WORK, sha256
    import control_queue, control_plan

SEEDS = (42, 43)
CONFIRMATION_SEEDS = (43,)
USER_INSTRUCTION = '2 seeds are enough, skip seed stuff'
CANCELLATION_SHA256 = '4fdad8030a05fcfda3c1f9af46442d9b57c92081e6d3f792ebc64a2f9969efcc'
read = control_queue.read


def destination():
    return WORK / 'protocol_amendments/two_seed_20261103'


def require_cancelled():
    target = destination()
    path = target / 'cancellation_complete.json'
    if sha256(path) != CANCELLATION_SHA256:
        raise ValueError('User-directed cancellation receipt changed')
    done, intent = read(path), read(target / 'cancellation_intent.json')
    if (done['phase'] != 'seed44_cancelled_for_user_two_seed_amendment' or
        done['intent_sha256'] != sha256(target / 'cancellation_intent.json') or
        intent['user_instruction'] != USER_INSTRUCTION or intent['seeds'] != list(SEEDS) or
        intent['skipped_seed'] != 44 or not done['B44_not_started'] or
        not done['partial_A44_preserved'] or done['completed_fits'] != 8):
        raise ValueError('Cancellation does not establish the requested two-seed scope')
    for identity in intent['stop_processes']:
        if control_queue.process_identity(identity['pid']) == identity:
            raise RuntimeError('Retired three-seed process is still live')
    if (WORK / 'controlled/runs/full__B__seed44__lr1p0').exists():
        raise RuntimeError('Skipped B seed44 output unexpectedly exists')
    jobs = [read(WORK / 'controlled/jobs' / (r['job_id'] + '.json'))
            for r in done['completed_evidence']]
    if (len(jobs) != 8 or any(j['seed'] not in SEEDS for j in jobs) or
        control_queue.require_completed(jobs) != done['completed_evidence']):
        raise ValueError('Completed two-seed evidence changed')
    return done


def build_policy():
    require_cancelled()
    primary_path = WORK / 'controlled/plan.json'
    primary = read(primary_path)
    if primary['tuning_seed'] != 42 or primary['confirmation_seeds'] != [43, 44]:
        raise ValueError('Unexpected historical seed registration')
    settings = len(primary['arms'])
    return dict(format_version=1, user_instruction=USER_INSTRUCTION,
        seeds=list(SEEDS), tuning_seed=42, confirmation_seeds=list(CONFIRMATION_SEEDS),
        skipped_seed=44, primary_plan_sha256=sha256(primary_path),
        original_spec_sha256=primary['spec_sha256'],
        cancellation_sha256=CANCELLATION_SHA256,
        source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))},
        primary_accounting=dict(settings=settings, tuning_runs=6 * settings,
            confirmation_runs=2 * settings, total_runs=8 * settings),
        scope='All remaining experiments use seeds42/43 only. This overrides the MD minimum of three training seeds; it does not remove experiments, LR candidates, split realizations or adaptation/test units.',
        interpretation='Seed42 is tuning-exposed; seed43 is the only confirmation seed. No confirmation-seed variance can be estimated. Partial A44 is excluded from selection and scientific comparisons; B44 is skipped.',
        historical_plans_unchanged=True, new_training_authorized_by_this_manifest=False)


def prepare():
    path = destination() / 'policy.json'
    control_queue.immutable(path, build_policy())
    return path


def verify():
    path = destination() / 'policy.json'
    record = read(path)
    if record != build_policy():
        raise ValueError('Two-seed policy or its bound inputs changed')
    return record


def require_allowed_job(job):
    if job['seed'] not in SEEDS or job['phase'] != ('tuning' if job['seed'] == 42 else 'confirmation'):
        raise ValueError('Job is outside the user-approved seed42/43 scope')


def confirm(arm_name):
    """Materialize only seed43 at the original validation-only chosen LR."""
    verify()
    directory = WORK / 'controlled'
    primary = read(directory / 'plan.json')
    digest = sha256(directory / 'plan.json')
    arm = next(a for a in primary['arms'] if a['name'] == arm_name)
    jobs = [j for j in primary['tuning_jobs'] if j['arm'] == arm_name]
    if len(jobs) != 6:
        raise ValueError('Expected six registered tuning fits')
    candidates, evidence = [], []
    for job in jobs:
        metadata = read(directory / 'jobs' / (job['id'] + '.json'))
        require_allowed_job(metadata)
        if metadata['plan_sha256'] != digest:
            raise ValueError('Tuning job belongs to another primary plan')
        control_queue.require_completed([metadata])
        candidate, source = control_plan.validation_candidate(metadata)
        evidence.append(source)
        if metadata['regime'] == 'B':
            candidates.append(candidate)
    selected = dict(arm=arm_name, plan_sha256=digest,
        **control_plan.select_learning_rate(candidates), evidence=evidence)
    selection_path = directory / 'selections' / (arm_name + '.json')
    control_queue.immutable(selection_path, selected)
    confirmed = []
    for regime in ('A', 'B'):
        multiplier = selected['learning_rate_multiplier']
        identity = control_plan.job_id(arm_name, regime, 43, multiplier)
        config = control_plan.make_config(primary['bases'][regime], arm, regime=regime,
            seed=43, multiplier=multiplier, output=directory / 'runs' / identity,
            training_noise_rms=primary['training_noise_rms'])
        job = dict(id=identity, arm=arm_name, regime=regime, seed=43,
            learning_rate_multiplier=multiplier, phase='confirmation', config=config,
            selection_sha256=sha256(selection_path))
        confirmed.append(control_plan.materialize_job(job, directory, digest))
    return dict(selection=selected, confirmation_jobs=confirmed)
