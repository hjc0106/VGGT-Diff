import hashlib
import json
from pathlib import Path


def test_exported_sources_match_the_recorded_export_hashes():
    root=Path(__file__).parents[1]
    rows=json.loads((root/'SOURCE_INDEX.json').read_text())['files']
    assert len(rows)>=100
    for row in rows:
        assert hashlib.sha256((root/row['destination']).read_bytes()).hexdigest()==row['export_sha256'],row['destination']
