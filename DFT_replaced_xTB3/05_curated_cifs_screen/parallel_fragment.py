#!/usr/bin/env python3
"""Run unchanged deCOFpose fragment command concurrently, one CIF per process."""
import argparse, csv, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--cif-dir',type=Path,required=True)
p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--script',type=Path,required=True)
p.add_argument('--pkgdir',type=Path,required=True);p.add_argument('--jobs',type=int,default=8)
a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
cifs=sorted(a.cif_dir.glob('*.cif'))
def one(cif):
    if all((a.out_dir/cif.stem/f'{x}.xyz').exists() for x in ('node','linker','nodelinker')):
        return cif.name,'cached'
    with tempfile.TemporaryDirectory(prefix='xtb3_frag_') as tmp:
        inp=Path(tmp)/'input';inp.mkdir();(inp/cif.name).symlink_to(cif.resolve())
        proc=subprocess.run([sys.executable,str(a.script),'fragment','--cif-dir',str(inp),
            '--out-dir',str(a.out_dir),'--pkgdir',str(a.pkgdir)],capture_output=True,text=True)
    ok=all((a.out_dir/cif.stem/f'{x}.xyz').exists() for x in ('node','linker','nodelinker'))
    return cif.name,'ok' if ok else 'failed'
records=[]
with ThreadPoolExecutor(max_workers=a.jobs) as pool:
    futures={pool.submit(one,c):c for c in cifs}
    for i,f in enumerate(as_completed(futures),1):
        records.append(f.result())
        if i%25==0 or i==len(cifs):print(f'{i}/{len(cifs)}',flush=True)
with (a.out_dir/'parallel_manifest.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['file','status']);w.writerows(sorted(records))
print(dict((s,sum(x[1]==s for x in records)) for s in set(x[1] for x in records)))
