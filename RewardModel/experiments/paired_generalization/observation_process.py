"""Q22: distinguish recorded ChEMBL curation from unobserved pair selection.

Read-only archive audit, conditional on assays represented in the curated
dataset. Missing/censored activity records are never converted to negatives.
The archive's provenance is not a record of original inclusion probabilities.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .replicate_audit import KEY
else:
    from run import ROOT, WORK, sha256, write_json
    from replicate_audit import KEY


ASSAY_QUERY = """
SELECT a.assay_id AS raw_assay_id, a.chembl_id AS assay_id,
       td.chembl_id AS target_chembl_id, a.src_id AS assay_src_id,
       s.src_description AS assay_source, a.assay_test_type, a.assay_category,
       a.assay_type, a.confidence_score, a.relationship_type, a.bao_format,
       a.variant_id, d.doc_id, d.chembl_id AS doc_chembl_id,
       d.doc_type, d.year AS doc_year
FROM wanted_assays w JOIN assays a ON a.chembl_id=w.chembl_id
JOIN target_dictionary td ON td.tid=a.tid
LEFT JOIN source s ON s.src_id=a.src_id
LEFT JOIN docs d ON d.doc_id=a.doc_id
ORDER BY a.chembl_id
"""

CONTEXT_QUERY = """
SELECT act.assay_id AS raw_assay_id, act.src_id AS activity_src_id,
       act.standard_relation, act.standard_type, act.data_validity_comment,
       act.potential_duplicate,
       CASE WHEN kept.activity_id IS NOT NULL THEN 1 ELSE 0 END AS retained,
       CASE WHEN act.pchembl_value IS NULL THEN 'missing'
            WHEN typeof(act.pchembl_value) IN ('integer','real')
                 AND abs(act.pchembl_value)<1e308 THEN 'defined'
            ELSE 'invalid' END AS pchembl_status,
       COUNT(*) AS records,
       SUM(CASE WHEN typeof(act.pchembl_value) IN ('integer','real')
                      AND abs(act.pchembl_value)<1e308 THEN 1 ELSE 0 END) AS defined_pchembl,
       SUM(CASE WHEN typeof(act.pchembl_value) IN ('integer','real')
                      AND abs(act.pchembl_value)<1e308 AND act.pchembl_value>=? THEN 1 ELSE 0 END) AS positive_ge6,
       SUM(CASE WHEN typeof(act.pchembl_value) IN ('integer','real')
                      AND abs(act.pchembl_value)<1e308 AND act.pchembl_value>? THEN 1 ELSE 0 END) AS active_gt6
FROM scoped_assays scope
CROSS JOIN activities act INDEXED BY fk_act_assay_id ON act.assay_id=scope.raw_assay_id
LEFT JOIN retained_measurements kept ON kept.activity_id=act.activity_id
GROUP BY act.assay_id, act.src_id, act.standard_relation, act.standard_type,
         act.data_validity_comment, act.potential_duplicate, retained, pchembl_status
"""


def read(path):
    return json.loads(Path(path).read_text())


def validate_inputs(source, retained, threshold=6.):
    if threshold != 6. or source.duplicated(KEY).any() or retained.activity_id.duplicated().any():
        raise ValueError('Expected unique source observations and original activity IDs')
    if not retained.verified.eq(True).all() or not np.issubdtype(retained.activity_id.dtype,np.integer):
        raise ValueError('Retained measurements are unverified or have invalid IDs')
    if not np.isfinite(source.pchembl_value).all() or not np.isfinite(retained.pchembl_value).all():
        raise ValueError('Original retained measurements must be finite')
    if not np.array_equal(source.binary_label,source.pchembl_value.ge(threshold).astype(int)):
        raise ValueError('Source is not the original per-assay threshold-labelled dataset')
    stats = retained.groupby(KEY,sort=True).pchembl_value.agg(['size','median'])
    expected = source.set_index(KEY).sort_index()
    if (not stats.index.equals(expected.index) or
        not np.array_equal(stats['size'],expected.measurement_count) or
        not np.allclose(stats['median'],expected.pchembl_value,rtol=0,atol=1e-8)):
        raise ValueError('Retained activity IDs do not reconstruct the source observations')


def extract_context(connection, source, retained, threshold=6.):
    """Only temporary tables are written; main database must be opened read-only."""
    validate_inputs(source,retained,threshold)
    if source.groupby('assay_id').target_chembl_id.nunique().max()!=1:
        raise ValueError('Curated assay has ambiguous target identity')
    connection.execute('PRAGMA temp_store=MEMORY')
    connection.execute('CREATE TEMP TABLE wanted_assays (chembl_id TEXT PRIMARY KEY)')
    connection.executemany('INSERT INTO wanted_assays VALUES (?)',((x,) for x in sorted(source.assay_id.unique())))
    assays = pd.read_sql_query(ASSAY_QUERY,connection)
    expected = source[['assay_id','target_chembl_id']].drop_duplicates().sort_values('assay_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(assays[['assay_id','target_chembl_id']],expected,check_dtype=False)
    if assays.raw_assay_id.duplicated().any():
        raise ValueError('Raw assay mapping is not unique')
    connection.execute('CREATE TEMP TABLE scoped_assays (raw_assay_id INTEGER PRIMARY KEY)')
    connection.executemany('INSERT INTO scoped_assays VALUES (?)',((int(x),) for x in assays.raw_assay_id))
    connection.execute('CREATE TEMP TABLE retained_measurements (activity_id INTEGER PRIMARY KEY, assay_id TEXT, pchembl_value REAL)')
    connection.executemany('INSERT INTO retained_measurements VALUES (?,?,?)',
        retained[['activity_id','assay_id','pchembl_value']].itertuples(index=False,name=None))
    mismatches = connection.execute("""
      SELECT COUNT(*) FROM retained_measurements kept
      LEFT JOIN activities act ON act.activity_id=kept.activity_id
      LEFT JOIN assays a ON a.assay_id=act.assay_id
      WHERE act.activity_id IS NULL OR a.chembl_id IS NULL OR a.chembl_id!=kept.assay_id
         OR act.pchembl_value IS NULL OR abs(act.pchembl_value-kept.pchembl_value)>1e-8
    """).fetchone()[0]
    if mismatches:
        raise ValueError('Retained activity IDs differ from the raw database')
    print(json.dumps(dict(stage='retained_ids_verified',measurements=len(retained),assays=len(assays))),flush=True)
    chunks=[]
    for chunk in pd.read_sql_query(CONTEXT_QUERY,connection,params=(threshold,threshold),chunksize=250000):
        chunks.append(chunk)
        print(json.dumps(dict(stage='raw_context',aggregate_rows=sum(len(c) for c in chunks))),flush=True)
    groups=pd.concat(chunks,ignore_index=True)
    if int(groups.loc[groups.retained.eq(1),'records'].sum())!=len(retained):
        raise ValueError('Raw assay context does not contain exactly the retained measurements')
    if not np.array_equal(groups.defined_pchembl,groups.records*groups.pchembl_status.eq('defined')):
        raise ValueError('Defined/missing activity accounting is inconsistent')
    sources=pd.read_sql_query('SELECT src_id,src_description FROM source',connection)
    return assays,groups,sources


def count_summary(groups):
    total=int(groups.records.sum());defined=int(groups.defined_pchembl.sum())
    positive=int(groups.positive_ge6.sum());active=int(groups.active_gt6.sum())
    return dict(records=total,defined_pchembl=defined,missing_or_invalid_pchembl=total-defined,
        positive_ge6=positive,active_gt6=active,
        positive_fraction_among_defined=positive/defined if defined else None,
        active_fraction_among_defined=active/defined if defined else None)


def summarize(source, retained, assays, groups, sources):
    # These mutually exclusive reasons describe our curation, not the decision
    # to test a molecule in the originating laboratory or publish its result.
    reasons=np.select([
        groups.retained.eq(1),groups.pchembl_status.ne('defined'),
        groups.standard_relation.fillna('').ne('='),
        groups.data_validity_comment.notna() & groups.data_validity_comment.fillna('').ne('Manually validated')],
        ['retained','missing_or_invalid_pchembl','nonexact_or_missing_relation','questionable_validity'],
        default='remaining_join_structure_or_encoding_exclusions')
    groups=groups.assign(curation_reason=reasons)
    if (groups.retained.eq(1)&(groups.pchembl_status.ne('defined')|groups.standard_relation.fillna('').ne('=')|
        (groups.data_validity_comment.notna()&groups.data_validity_comment.fillna('').ne('Manually validated')))).any():
        raise ValueError('Retained records violate the recorded exact/defined filters')
    source_counts=source.assign(positive_ge6=source.binary_label,active_gt6=source.pchembl_value.gt(6.).astype(int))
    by_assay=source_counts.groupby('assay_id').agg(curated_observations=('compound_id','size'),
        curated_positive_ge6=('positive_ge6','sum'),curated_active_gt6=('active_gt6','sum'))
    assays=assays.merge(by_assay,on='assay_id',validate='one_to_one')
    categories={}
    for name in ('assay_source','assay_test_type','assay_category','doc_type','doc_year'):
        table=assays.groupby(name,dropna=False).agg(assays=('assay_id','size'),documents=('doc_id','nunique'),
            curated_observations=('curated_observations','sum'),curated_positive_ge6=('curated_positive_ge6','sum'))
        table['curated_positive_fraction']=table.curated_positive_ge6/table.curated_observations
        categories[name]=json.loads(table.reset_index().to_json(orient='records'))
    totals=dict(all_archived_records_in_retained_assays=count_summary(groups),
        retained_measurements=count_summary(groups[groups.retained.eq(1)]),
        excluded_records_in_retained_assays=count_summary(groups[groups.retained.eq(0)]))
    source_names=dict(zip(sources.src_id,sources.src_description))
    activity_sources=[dict(src_id=int(key[0]) if pd.notna(key[0]) else None,source=source_names.get(key[0]),retained=bool(key[1]),**count_summary(g))
        for key,g in groups.groupby(['activity_src_id','retained'],dropna=False,sort=True)]
    summary=dict(question='Q22',scope='All archived activity records in assays with at least one retained curated observation; not all ChEMBL assays or all originally screened pairs',
        thresholds=dict(curated_binary_label='pChEMBL >=6 before pair-median harmonization',contrastive_active='pChEMBL >6'),
        curated=dict(observations=len(source),assays=len(assays),documents=int(assays.doc_id.nunique()),
            proteins=int(source.target_chembl_id.nunique()),molecules=int(source.compound_id.nunique()),
            positive_observations=int(source.binary_label.sum()),positive_fraction=float(source.binary_label.mean())),
        totals=totals,curation_reasons={str(k):count_summary(g) for k,g in groups.groupby('curation_reason',sort=True)},
        provenance_categories=categories,activity_sources=activity_sources,
        interpretation='The exact-measurement/quality/structure filters and median aggregation are observable. Source/test-type annotations describe provenance, not individual pair inclusion probabilities. Missing and censored records remain unlabeled here. Their frequencies cannot identify the activity prevalence of unobserved or generator-produced molecules, or prove that promising pairs were preferentially tested.',
        unresolved='Original experiment sampling frames, all tested-but-unpublished pairs, per-pair inclusion probabilities and the reason each pair was selected are not reconstructed by these database tables.')
    return summary,assays,groups


def main():
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Run the data audit CPU-only in the authorized Atum project')
    output=WORK/'tier1/observation_process'
    if output.exists():
        raise RuntimeError('Inspect existing observation-process artifacts before rerunning')
    original=ROOT/'dataset/processed/chembl_37/protein_cluster_50_activity_balanced'
    raw=WORK/'tier1/replicates'
    source_info=read(original/'summary.json');reconstruction=read(raw/'reconstruction.json')
    download=read(ROOT/'dataset/raw/chembl_37/download_verified.json')
    if (reconstruction['unverified_source_rows']!=0 or reconstruction['source_sha256']!=sha256(original/'all.parquet') or
        source_info['source']['sqlite_archive_sha256']!=download['sha256'] or download['sha256']!=reconstruction['archive_sha256']):
        raise ValueError('Original dataset/archive reconstruction provenance changed')
    expected=dict(assay_type='B',target_type='SINGLE PROTEIN',exact_standard_relation_only=True,
        exactly_one_protein_component=True,exclude_questionable_data_validity=True,exclude_variants=True,
        pchembl_value_not_null=True,min_confidence_score=8,activity_threshold=6.)
    if any(source_info['filters'].get(k)!=v for k,v in expected.items()):
        raise ValueError('Unexpected original curation policy')
    source=pd.read_parquet(original/'all.parquet',columns=[*KEY,'pchembl_value','measurement_count','binary_label'])
    retained=pd.read_parquet(raw/'raw_measurements.parquet')
    database=ROOT/'dataset/raw/chembl_37/chembl_37.db'
    before=database.stat()
    print(json.dumps(dict(stage='hashing_raw_database',bytes=before.st_size)),flush=True)
    database_hash=sha256(database)
    connection=sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True)
    try:
        schema={name:[r[1] for r in connection.execute(f'PRAGMA table_info({name})')]
            for name in ('activities','assays','docs','source')}
        assays,groups,sources=extract_context(connection,source,retained)
        if (not assays.assay_type.eq('B').all() or not assays.confidence_score.ge(8).all() or
            assays.variant_id.notna().any()):
            raise ValueError('Retained assay annotations violate the original curation filters')
        # Independent SQL totals bypass the grouped accounting and membership join.
        totals=connection.execute("""SELECT COUNT(*),COUNT(act.pchembl_value)
            FROM scoped_assays scope CROSS JOIN activities act INDEXED BY fk_act_assay_id
            ON act.assay_id=scope.raw_assay_id""").fetchone()
        if totals[0]!=int(groups.records.sum()):
            raise ValueError('Independent raw activity count differs')
    finally:
        connection.close()
    after=database.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('Raw database changed during the read-only audit')
    summary,assays,groups=summarize(source,retained,assays,groups,sources)
    output.mkdir(parents=True)
    assays.to_parquet(output/'assays.parquet',index=False)
    groups.to_parquet(output/'activity_context_groups.parquet',index=False)
    sources.to_parquet(output/'sources.parquet',index=False)
    (output/'context.sql').write_text(CONTEXT_QUERY)
    (output/'assays.sql').write_text(ASSAY_QUERY)
    code=[Path(__file__),Path(__file__).with_name('replicate_audit.py'),Path(__file__).with_name('run.py'),ROOT/'data_processing/build_chembl_binding_dataset.py']
    provenance=dict(database_sha256=database_hash,database_bytes=before.st_size,archive_sha256=download['sha256'],
        source_sha256=sha256(original/'all.parquet'),source_summary_sha256=sha256(original/'summary.json'),
        reconstruction_sha256=sha256(raw/'reconstruction.json'),retained_measurements_sha256=sha256(raw/'raw_measurements.parquet'),
        source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in code},sqlite_version=sqlite3.sqlite_version,
        pandas_version=pd.__version__,numpy_version=np.__version__,schema=schema,
        original_filters=source_info['filters'],original_curation_counts=source_info['curation'],
        independent_sql_records=int(totals[0]),independent_sql_nonnull_pchembl=int(totals[1]),
        database_unchanged=True,training_data_modified=False,test_predictions_read=False)
    write_json(output/'complete.json',dict(provenance=provenance,summary=summary,
        artifacts_sha256={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(stage='observation_process_complete',curated=summary['curated'],totals=summary['totals'])),flush=True)


if __name__=='__main__':
    main()
