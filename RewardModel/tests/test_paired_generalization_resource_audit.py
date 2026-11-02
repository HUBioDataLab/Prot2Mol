"""Reference-paced estimates stay separate from observed costs and scope."""
import os

import pytest

from experiments.paired_generalization import resource_audit as module


def test_cadence_uses_only_observed_inter_save_intervals():
    samples=[dict(epoch=i,mtime=t) for i,t in enumerate([1000.,1100.,1220.,1325.],1)]
    result=module.cadence(samples)
    assert result['seconds_between_saves']==dict(minimum=100.,median=105.,maximum=120.)
    assert result['horizon_seconds_at_median_cadence']==1575.
    assert result['observed_first_to_last_save_seconds']==325.
    assert result['completed_epochs']==4


@pytest.mark.parametrize('samples',[[dict(epoch=1,mtime=1.)],
    [dict(epoch=1,mtime=1.),dict(epoch=3,mtime=3.)],
    [dict(epoch=1,mtime=2.),dict(epoch=2,mtime=2.)],
    [dict(epoch=1,mtime=2.),dict(epoch=2,mtime=float('nan'))]])
def test_incomplete_or_invalid_save_timings_are_rejected(samples):
    with pytest.raises(ValueError):module.cadence(samples)


def test_allocated_storage_deduplicates_hard_links_and_skips_symlinks(tmp_path):
    first=tmp_path/'run';second=tmp_path/'protected';first.mkdir();second.mkdir()
    weight=first/'pytorch_model.bin';weight.write_bytes(b'weight bytes'*1024)
    os.link(weight,second/'pytorch_model.bin');(second/'alias.bin').symlink_to(weight)
    optimizer=first/'optimizer.pt';optimizer.write_bytes(b'optimizer bytes'*512)
    records=module.footprint([first,second])
    assert records[0]['files']==2 and records[1]['files']==1
    assert records[1]['logical_bytes']==weight.stat().st_size
    assert records[1]['additional_allocated_bytes_across_roots']==0
    assert records[0]['unique_allocated_bytes']==weight.stat().st_blocks*512+optimizer.stat().st_blocks*512


def test_projection_keeps_primary_and_optional_counts_distinct():
    primary=dict(arms=[dict(name=str(i)) for i in range(28)],confirmation_seeds=[43,44],
        tuning_jobs=[dict(regime=r) for _ in range(84) for r in ('A','B')],
        accounting=dict(total_runs=280,epochs_per_run=15))
    result=module.scenario(primary,dict(A=dict(horizon_seconds_at_median_cadence=21600),B=dict(horizon_seconds_at_median_cadence=25200)),500,4)
    assert result['primary_fit_counts']==dict(A=140,B=140)
    assert result['full_model_pace_primary_wall_days']==pytest.approx(140*13/24)
    assert result['full_model_pace_primary_gpu_hours']==140*13*4
    assert result['retained_full_weight_bytes']==140*(4+17)*500
    assert result['additional_allocation_confirmation_bound']['maximum_fits_all_settings']==112
    assert result['additional_allocation_confirmation_bound']['minimum_fits']==0
    assert 'not a completion ETA' in result['interpretation']
    primary['accounting']['total_runs']=281
    with pytest.raises(ValueError):module.scenario(primary,{},500,4)
