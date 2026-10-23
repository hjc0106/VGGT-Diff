import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('migration_relocate',Path(__file__).parents[1]/'relocate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_prefix_mapping_does_not_rewrite_sibling_or_prose():
    assert m.map_string('/old/data/a',{'/old/data':'/new'})=='/new/a'
    assert m.map_string('/old/database/a',{'/old/data':'/new'})=='/old/database/a'
    assert m.map_string('text /old/data/a',{'/old/data':'/new'})=='text /old/data/a'


def test_relocation_dry_run_apply_hashes_and_backup(tmp_path):
    workspace=tmp_path/'new';data=workspace/'pipeline_exp';data.mkdir(parents=True)
    image=data/'image.png';image.write_bytes(b'image pixels unchanged')
    manifest=data/'manifest.json';old='/old/server/project'
    expected=hashlib.sha256(image.read_bytes()).hexdigest()
    original={'frozen_files':[{'path':old+'/pipeline_exp/image.png','sha256':expected}]}
    manifest.write_text(json.dumps(original))
    preview=m.relocate(data,old,workspace=workspace)
    assert not preview['applied'] and not preview['missing_assets']
    assert json.loads(manifest.read_text())==original
    applied=m.relocate(data,old,workspace=workspace,apply=True)
    row=json.loads(manifest.read_text())['frozen_files'][0]
    assert row['path']==str(image) and row['sha256']==expected
    assert (Path(applied['backup'])/'pipeline_exp/manifest.json').exists()
    assert image.read_bytes()==b'image pixels unchanged'


def test_missing_assets_prevent_any_writes(tmp_path):
    data=tmp_path/'pipeline_exp';data.mkdir()
    p=data/'manifest.json';original={'frozen_files':[{'path':'/old/missing.png','sha256':'bad'}]}
    p.write_text(json.dumps(original))
    result=m.relocate(data,'/old',workspace=tmp_path,apply=True)
    assert result['missing_assets'] and not result['applied']
    assert json.loads(p.read_text())==original


def test_original_workspace_cannot_be_rewritten(tmp_path):
    with pytest.raises(ValueError):m.relocate(tmp_path,str(tmp_path),workspace=tmp_path)


def test_manifest_and_downstream_digest_references_follow_relocated_json(tmp_path):
    data=tmp_path/'pipeline_exp';data.mkdir()
    image=data/'image.png';image.write_bytes(b'pixels')
    protocol=data/'protocol.json';protocol.write_text(json.dumps({'image':'/old/pipeline_exp/image.png'}))
    old_digest=m.sha(protocol)
    manifest=data/'manifest.json';manifest.write_text(json.dumps({'frozen_files':[{'path':'/old/pipeline_exp/protocol.json','sha256':old_digest}]}))
    review=data/'review.json';review.write_text(json.dumps({'protocol_sha256':old_digest}))
    result=m.relocate(data,'/old',workspace=tmp_path,apply=True)
    assert result['applied']
    assert json.loads(manifest.read_text())['frozen_files'][0]['sha256']==m.sha(protocol)
    assert json.loads(review.read_text())['protocol_sha256']==m.sha(protocol)
