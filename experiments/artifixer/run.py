"""Run one archived script with the expected workspace and import paths."""
import os
from pathlib import Path
import runpy
import sys

WORKSPACE=Path(__file__).resolve().parent/'workspace'


def main():
    if len(sys.argv)<2 or sys.argv[1]=='--list':
        print('\n'.join(p.name for p in sorted((WORKSPACE/'scripts').glob('*.py'))))
        return
    script=Path(sys.argv[1])
    if script.name!=str(script) or script.suffix!='.py':
        raise SystemExit('Use a script basename from --list')
    path=WORKSPACE/'scripts'/script
    if not path.is_file():raise SystemExit(f'Unknown script: {script}')
    sys.path[:0]=[str(WORKSPACE/'scripts'),str(WORKSPACE),str(WORKSPACE/'worldact'),
        str(WORKSPACE/'thirdparty/VGGT-Diff'),str(WORKSPACE/'thirdparty/3DGRUT-ArtiFixer'),
        str(WORKSPACE/'thirdparty/2d-gaussian-splatting')]
    os.chdir(WORKSPACE)
    sys.argv=[str(path)]+sys.argv[2:]
    runpy.run_path(str(path),run_name='__main__')


if __name__=='__main__':main()
