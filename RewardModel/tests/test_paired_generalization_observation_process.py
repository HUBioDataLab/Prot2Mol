import hashlib
import sqlite3

import pandas as pd
import pytest

from experiments.paired_generalization import observation_process as audit


def fixture(tmp_path):
    path=tmp_path/'raw.db';connection=sqlite3.connect(path)
    connection.executescript('''
      CREATE TABLE assays(assay_id INTEGER PRIMARY KEY,chembl_id TEXT,tid INTEGER,src_id INTEGER,
        assay_test_type TEXT,assay_category TEXT,assay_type TEXT,confidence_score INTEGER,
        relationship_type TEXT,bao_format TEXT,variant_id INTEGER,doc_id INTEGER);
      CREATE TABLE activities(activity_id INTEGER PRIMARY KEY,assay_id INTEGER,src_id INTEGER,
        standard_relation TEXT,standard_type TEXT,data_validity_comment TEXT,potential_duplicate INTEGER,pchembl_value REAL);
      CREATE INDEX fk_act_assay_id ON activities(assay_id);
      CREATE TABLE target_dictionary(tid INTEGER PRIMARY KEY,chembl_id TEXT);
      CREATE TABLE source(src_id INTEGER PRIMARY KEY,src_description TEXT);
      CREATE TABLE docs(doc_id INTEGER PRIMARY KEY,chembl_id TEXT,doc_type TEXT,year INTEGER);
      INSERT INTO target_dictionary VALUES(1,'p1'),(2,'p2');
      INSERT INTO source VALUES(1,'Scientific Literature'),(2,'Deposited Data');
      INSERT INTO docs VALUES(1,'doc1','PUBLICATION',2020),(2,'doc2','DATASET',2021);
      INSERT INTO assays VALUES(1,'a1',1,1,'Confirmatory',NULL,'B',9,'D','bio',NULL,1),
        (2,'a2',2,2,NULL,NULL,'B',8,'D','bio',NULL,2),
        (3,'outside',1,1,NULL,NULL,'B',9,'D','bio',NULL,1);
    ''')
    rows=[(1,1,1,'=','IC50',None,0,8.),(2,1,1,'=','IC50',None,0,4.),
          (3,1,1,'>','IC50',None,0,None),(4,1,None,'=','Inhibition',None,0,None),
          (5,1,1,'=','IC50','Outside typical range',0,7.),
          (6,1,1,'<','IC50',None,0,5.),(7,1,1,'=','IC50',None,0,6.),
          (8,2,2,'=','Kd','Manually validated',0,5.),(9,2,2,'=','Kd',None,0,7.),
          (10,3,1,'=','Kd',None,0,9.)]
    connection.executemany('INSERT INTO activities VALUES(?,?,?,?,?,?,?,?)',rows)
    connection.commit();connection.close()
    source=pd.DataFrame(dict(assay_id=['a1','a1','a1','a2'],target_chembl_id=['p1','p1','p1','p2'],
        compound_id=['m1','m2','m3','m4'],pchembl_value=[8.,4.,6.,5.],measurement_count=[1]*4,binary_label=[1,0,1,0]))
    retained=source.drop(columns=['measurement_count','binary_label']).assign(activity_id=[1,2,7,8],verified=True)
    return path,source,retained


def test_read_only_audit_separates_curation_from_missing_and_censored_measurements(tmp_path):
    path,source,retained=fixture(tmp_path)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as connection:
        assays,groups,sources=audit.extract_context(connection,source,retained)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    result,assays,groups=audit.summarize(source,retained,assays,groups,sources)
    assert len(assays)==2 and set(assays.assay_id)=={'a1','a2'}
    assert result['curated']['observations']==4 and result['curated']['positive_fraction']==.5
    raw=result['totals']['all_archived_records_in_retained_assays']
    assert raw['records']==9 and raw['defined_pchembl']==7
    assert raw['missing_or_invalid_pchembl']==2
    assert raw['positive_fraction_among_defined']==pytest.approx(4/7)
    assert raw['active_fraction_among_defined']==pytest.approx(3/7)
    assert {key:value['records'] for key,value in result['curation_reasons'].items()}=={
        'retained':4,'missing_or_invalid_pchembl':2,'nonexact_or_missing_relation':1,
        'questionable_validity':1,'remaining_join_structure_or_encoding_exclusions':1}
    kept=result['totals']['retained_measurements']
    assert kept['positive_ge6']==2 and kept['active_gt6']==1
    # Missing values were not assigned a negative binary label.
    missing=groups[groups.pchembl_status.eq('missing')]
    assert missing.positive_ge6.sum()==0 and missing.defined_pchembl.sum()==0
    assert not any(column in groups for column in ['binary_label','negative_label'])
    assert sum(x['curated_observations'] for x in result['provenance_categories']['assay_source'])==4
    assert sum(x['records'] for x in result['activity_sources'])==9
    assert next(x for x in result['activity_sources'] if x['src_id'] is None)['records']==1


def test_replicate_counts_and_exact_medians_must_reconstruct_source(tmp_path):
    _,source,retained=fixture(tmp_path)
    changed=source.copy();changed.loc[0,'measurement_count']=2
    with pytest.raises(ValueError,match='reconstruct'):
        audit.validate_inputs(changed,retained)
    changed=retained.copy();changed.loc[0,'pchembl_value']=7.5
    with pytest.raises(ValueError,match='reconstruct'):
        audit.validate_inputs(source,changed)
    changed=retained.copy();changed.loc[0,'verified']=False
    with pytest.raises(ValueError,match='unverified'):
        audit.validate_inputs(source,changed)
    with pytest.raises(ValueError,match='unique source'):
        audit.validate_inputs(source,pd.concat([retained,retained.iloc[:1]],ignore_index=True))


def test_saved_activity_ids_cannot_refer_to_different_database_values(tmp_path):
    path,source,retained=fixture(tmp_path)
    # Keep the source/raw-Parquet reconstruction internally consistent, but
    # change one source value away from the actual released database.
    source.loc[0,'pchembl_value']=7.5;retained.loc[0,'pchembl_value']=7.5
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as connection:
        with pytest.raises(ValueError,match='differ from the raw database'):
            audit.extract_context(connection,source,retained)


def test_undefined_positive_fraction_is_not_zero_or_a_negative_class(tmp_path):
    _,source,retained=fixture(tmp_path)
    groups=pd.DataFrame(dict(records=[4],defined_pchembl=[0],positive_ge6=[0],active_gt6=[0]))
    result=audit.count_summary(groups)
    assert result['positive_fraction_among_defined'] is None
    assert result['active_fraction_among_defined'] is None
