"""List data to rsync separately from Git, including referenced external inputs."""
import argparse
from collections import deque
import json
from pathlib import Path


def values(obj):
    if isinstance(obj,str):yield obj
    elif isinstance(obj,list):
        for x in obj:yield from values(x)
    elif isinstance(obj,dict):
        for x in obj.values():yield from values(x)


def inventory(workspace,data_root):
    workspace=Path(workspace).resolve();data_root=Path(data_root).resolve()
    allowed=workspace/'pipeline_exp'
    if not data_root.is_relative_to(allowed):raise ValueError('Select a data root inside workspace/pipeline_exp')
    pending=deque(p for p in data_root.rglob('*') if p.is_file() and not p.is_symlink())
    found=set();missing=set()
    while pending:
        p=pending.popleft().resolve()
        if p in found or not p.is_relative_to(allowed):continue
        if not p.is_file():missing.add(str(p));continue
        found.add(p)
        if p.suffix!='.json':continue
        try:obj=json.loads(p.read_text())
        except (ValueError,UnicodeError):continue
        for s in values(obj):
            if not s.startswith('/'):continue
            q=Path(s)
            if q.is_relative_to(allowed) and q.is_file():pending.append(q)
    files=[dict(path=str(p.relative_to(workspace)),bytes=p.stat().st_size) for p in sorted(found)]
    return dict(files=files,total_bytes=sum(r['bytes'] for r in files),missing_files=sorted(missing),
                excludes='External model weights, caches, environments and credentials. Transfer model weights separately.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=inventory(args.workspace,args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    args.output.with_suffix('.files').write_text(''.join(r['path']+'\n' for r in result['files']))
    print(json.dumps(dict(files=len(result['files']),total_gib=result['total_bytes']/2**30,
                         list=str(args.output.with_suffix('.files'))),indent=2))


if __name__=='__main__':main()
