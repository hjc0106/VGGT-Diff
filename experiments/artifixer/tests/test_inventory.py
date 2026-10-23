import importlib.util
from pathlib import Path
import json

spec=importlib.util.spec_from_file_location('asset_inventory',Path(__file__).parents[1]/'asset_inventory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_inventory_follows_data_references_but_excludes_model_root(tmp_path):
    data=tmp_path/'pipeline_exp/run';data.mkdir(parents=True)
    shared=tmp_path/'pipeline_exp/shared.png';shared.write_bytes(b'pixels')
    model=tmp_path/'models/model.pt';model.parent.mkdir();model.write_bytes(b'weights')
    (data/'manifest.json').write_text(json.dumps({'image':str(shared),'checkpoint':str(model)}))
    result=m.inventory(tmp_path,data)
    paths={r['path'] for r in result['files']}
    assert 'pipeline_exp/shared.png' in paths
    assert 'models/model.pt' not in paths
