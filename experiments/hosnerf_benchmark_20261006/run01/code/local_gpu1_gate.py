#!/usr/bin/env python3
"""Wait for existing GPU1 evaluation receipts, then enter the frozen local queue.

This admission helper updates no scientific source and performs no model checks
or optimization of its own. File completion wakes it through Linux inotify.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import a100_worker
import local_gpu1_queue as queue

RUN = Path(__file__).resolve().parents[1]
OPS = RUN / 'local_gpu1'
PATHS = [queue.EXTERNAL / f'evaluation/r2_{arm}/async_complete.json' for arm in ['G', 'RG']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cutoff', default='2026-11-04T23:00:00-08:00')
    args = parser.parse_args()
    assert queue.sha(RUN / 'code/a100_worker.py') == 'b7f65044a25b8543300b603bbb997fdd9e8f47216c9e7c0706bbe8122d4d55b1'
    assert queue.sha(RUN / 'code/local_gpu1_queue.py') == '5e226d82606989748f141cf684777cf4a319979a25a9273f10c0547e6037209c'
    ops = queue.prepare('1')
    queue.assignment()
    waiter = a100_worker.Worker(args)
    if not args.run:
        # A real creation event exercises the already audited inotify helper.
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'receipt.json'
            child = subprocess.Popen(['/usr/bin/python3', '-c',
                'import pathlib,sys,time;time.sleep(.2);pathlib.Path(sys.argv[1]).write_text("{}")', str(target)])
            waiter.wait_file_event(target)
            assert child.wait(timeout=5) == 0 and json.loads(target.read_text()) == {}
        proof = dict(status='passed', gpu_tasks_launched=0, gate_source_sha256=queue.sha(Path(__file__)),
            wrapper_source_sha256=queue.sha(RUN / 'code/local_gpu1_queue.py'),
            event_helper_source_sha256=queue.sha(RUN / 'code/a100_worker.py'),
            real_file_creation_event_checked=True, required_receipts=[str(path) for path in PATHS])
        queue.write(ops / 'gate_cpu_acceptance.json', proof)
        print(json.dumps(proof), flush=True)
        return
    try:
        for path in PATHS:
            while not path.is_file():
                waiter.deadline_check()
                queue.write(ops / 'pipeline.json', dict(status='waiting_for_existing_4dsr_evaluation',
                    missing_receipt=str(path), worker_pid=os.getpid(), updated_unix=time.time()))
                waiter.wait_file_event(path)
        # Existing records must pass complete checkpoint and metric hash checks.
        queue.external_clearance()
        queue.write(ops / 'gate_clearance.json', dict(status='passed',
            receipts={str(path): queue.sha(path) for path in PATHS}, checked_unix=time.time()))
        args.gpu = '1'
        queue.run(args, ops)
    except a100_worker.DeadlineReached:
        queue.write(ops / 'terminal_receipt.json', dict(status='stopped_at_freeze_deadline',
            scenes=queue.OWNERS['1'], gpu='1', ended_unix=time.time(), gate_only=True))
    except BaseException:
        queue.write(ops / 'gate_failure.json', dict(status='failed', traceback=traceback.format_exc(),
            updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
