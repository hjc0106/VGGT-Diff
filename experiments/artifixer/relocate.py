"""Rebase a COPY of experiment JSON paths/hashes after data transfer.

Default is a dry run. --apply preserves original JSON in migration_backups and
records a new lineage; it does not claim the migrated manifests are originals.
Weights and raw images are never modified. Missing assets prevent all writes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

WORKSPACE=Path(__file__).resolve().parent/'workspace'
sys.path.insert(0,str(WORKSPACE/'scripts'))
from experiment_paths import checkpoint_path, omega_path, base_model_path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def map_string(value,mappings):
    for old,new in sorted(mappings.items(),key=lambda r:len(r[0]),reverse=True):
        if value==old or value.startswith(old+'/'):return new+value[len(old):]
    return value


def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,list):
        for row in value:yield from strings(row)
    elif isinstance(value,dict):
        for row in value.values():yield from strings(row)


def transform(value,mappings,digests):
    if isinstance(value,str):
        return digests.get(value,map_string(value,mappings))
    if isinstance(value,list):return [transform(v,mappings,digests) for v in value]
    if isinstance(value,dict):return {k:transform(v,mappings,digests) for k,v in value.items()}
    return value


def relocate(data_root,old_workspace,workspace=WORKSPACE,extra_maps=None,apply=False):
    data_root=Path(data_root).resolve();workspace=Path(workspace).resolve()
    if not data_root.is_relative_to(workspace):raise ValueError('Data root must be inside the copied workspace')
    if str(workspace)==old_workspace.rstrip('/'):raise ValueError('Refusing to rewrite the original workspace')
    files=sorted(data_root.rglob('*.json'))
    if not files:raise ValueError('No transferred JSON data found')
    originals={p:p.read_bytes() for p in files}
    values={p:json.loads(v) for p,v in originals.items()}
    maps={old_workspace.rstrip('/'):str(workspace),**(extra_maps or {})}
    # Discover old model paths from transferred metadata, not this machine's paths.
    for value in values.values():
        for s in strings(value):
            if s.startswith('/') and s.endswith('/vggt-diff.safetensors'):
                maps[str(Path(s).parent)]=str(checkpoint_path().parent)
            elif s.startswith('/') and s.endswith('/vggt_omega_1b_512.pt'):
                maps[str(Path(s).parent)]=str(omega_path().parent)
            elif s.startswith('/') and Path(s).name=='Wan2.1-I2V-14B-480P':maps[s]=str(base_model_path())
    maps.update(extra_maps or {})
    digests={};cache={};missing=set()
    def current_hash(p):
        if p in values:return hashlib.sha256((json.dumps(values[p],indent=2)+'\n').encode()).hexdigest()
        st=p.stat();key=(p,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha(p)
        return cache[key]
    def refresh(value):
        if isinstance(value,list):
            for v in value:refresh(v)
        elif isinstance(value,dict):
            for key,candidates in [('sha256',['path','image','selected_gt_path','original','source']),
                ('mask_sha256',['mask']),('full_sha256',['full_path','full_image'])]:
                if key not in value:continue
                for candidate in candidates:
                    s=value.get(candidate)
                    if not isinstance(s,str) or not s.startswith('/'):continue
                    p=Path(s)
                    if not p.is_file():missing.add(s);continue
                    old=value[key];new=current_hash(p)
                    if isinstance(old,str) and old!=new:digests[old]=new
                    value[key]=new;break
            for row in value.get('weights',[]):
                if isinstance(row,dict) and isinstance(row.get('path'),str):
                    p=Path(row['path'])
                    if p.is_file():
                        st=p.stat();row['size']=st.st_size;row['mtime_ns']=st.st_mtime_ns
                    else:missing.add(str(p))
            for v in value.values():refresh(v)
    # JSON manifest DAGs may refer to one another; propagate changed digests.
    for iteration in range(40):
        before={p:json.dumps(v,sort_keys=True) for p,v in values.items()}
        for p in files:values[p]=transform(values[p],maps,digests)
        for p in files:refresh(values[p])
        for p in files:
            old=hashlib.sha256(originals[p]).hexdigest();new=current_hash(p)
            if old!=new:digests[old]=new
        if all(before[p]==json.dumps(values[p],sort_keys=True) for p in files):break
    else:raise ValueError('Digest references did not converge; no data written')
    report=dict(json_files=len(files),changed=sum(json.loads(originals[p])!=values[p] for p in files),
        missing_assets=sorted(missing),maps=maps,applied=False,
        lineage='Path relocation and code/weight metadata rebaseline, not original experiment hashes')
    if missing:return report
    if apply:
        backup=workspace/'migration_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        for p in files:
            b=backup/p.relative_to(workspace);b.parent.mkdir(parents=True,exist_ok=True);b.write_bytes(originals[p])
        for p in files:p.write_text(json.dumps(values[p],indent=2)+'\n')
        report.update(applied=True,backup=str(backup))
        (backup/'MIGRATION.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--old-workspace',required=True)
    p.add_argument('--map',action='append',default=[],metavar='OLD=NEW')
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    report=relocate(args.data_root,args.old_workspace,extra_maps=dict(s.split('=',1) for s in args.map),apply=args.apply)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['missing_assets']:raise SystemExit('Missing assets: no changes applied')


if __name__=='__main__':main()
