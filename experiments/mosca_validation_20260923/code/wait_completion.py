"""Wait for pipeline atomic status writes using inotify, without polling GPUs."""
import argparse, ctypes, json, os, select, struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('branch',type=Path);a=p.parse_args()
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0: raise OSError(ctypes.get_errno(),'inotify_init1')
wd=libc.inotify_add_watch(fd,os.fsencode(a.branch),0x00000080 | 0x00000008)
if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
try:
    while True:
        record=json.loads((a.branch/'pipeline.json').read_text())
        if record['status']!='running':
            print(json.dumps({'status':record['status'],'branch':str(a.branch),'wall_seconds':record.get('wall_seconds'),'stages':record['stages']},indent=2),flush=True)
            break
        select.select([fd],[],[])
        os.read(fd,65536)
finally:
    os.close(fd)
