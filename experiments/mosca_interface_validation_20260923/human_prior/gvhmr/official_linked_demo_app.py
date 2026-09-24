import os
import time
import subprocess
from pathlib import Path

REPO_ROOT = str(os.path.join(os.path.dirname(__file__)))

def run_cmds(cmds):
    cmds = cmds.split('\n')
    for cmd in cmds:
        if len(cmd) == 0:
            continue
        os.system(cmd)


def prepare_env():
    init_flag = Path(f'{REPO_ROOT}/initialized')
    if init_flag.exists():
        return

    os.chdir(REPO_ROOT)
    run_cmds(
        f'''
        git clone https://github.com/zju3dv/GVHMR --recursive {REPO_ROOT}/GVHMR
        pip install -r {REPO_ROOT}/GVHMR/requirements.txt
        pip install -e {REPO_ROOT}/GVHMR
        mkdir {REPO_ROOT}/GVHMR/inputs
        mkdir {REPO_ROOT}/GVHMR/outputs
        ln -s {REPO_ROOT}/GVHMR/inputs {REPO_ROOT}/
        ln -s {REPO_ROOT}/GVHMR/outputs {REPO_ROOT}/
        ln -s {REPO_ROOT}/GVHMR/hmr4d {REPO_ROOT}/
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl -d {REPO_ROOT}/GVHMR/inputs/checkpoints/body_models/smpl -o SMPL_NEUTRAL.pkl
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz -d {REPO_ROOT}/GVHMR/inputs/checkpoints/body_models/smplx -o SMPLX_NEUTRAL.npz
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/dpvo/dpvo.pth -d {REPO_ROOT}/GVHMR/inputs/checkpoints/dpvo -o dpvo.pth
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt -d {REPO_ROOT}/GVHMR/inputs/checkpoints/gvhmr -o gvhmr_siga24_release.ckpt
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt -d {REPO_ROOT}/GVHMR/inputs/checkpoints/hmr2 -o epoch=10-step=25000.ckpt
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth -d {REPO_ROOT}/GVHMR/inputs/checkpoints/vitpose -o vitpose-h-multi-coco.pth
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt -d {REPO_ROOT}/GVHMR/inputs/checkpoints/yolo -o yolov8x.pt
        touch {REPO_ROOT}/initialized
        '''
    )

    return


def server_up():
    server_up_flag = Path(f'{REPO_ROOT}/server_up')
    if server_up_flag.exists():
        while True:
            time.sleep(600)
    else:
        os.system(f'python {REPO_ROOT}/app/entry.py')


if __name__ == '__main__':
    prepare_env()
    server_up()