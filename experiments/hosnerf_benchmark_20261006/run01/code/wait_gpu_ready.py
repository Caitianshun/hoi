#!/usr/bin/env python3
"""Bounded, read-only NVIDIA driver admission for local systemd ExecStartPre.

This helper checks driver accessibility and fixed physical GPU identity only.
The frozen queue still owns its locks and its original double-read idle guard.
No CUDA context, model, optimization, driver mutation, or resource lease is used.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


IDENTITIES = {
    '0': dict(uuid='GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5',
              name='NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
              memory_total_mib=97887),
    '1': dict(uuid='GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec',
              name='NVIDIA GeForce RTX 3090', memory_total_mib=24576),
}
MAX_WAIT_SECONDS = 600.0
QUERY_TIMEOUT_SECONDS = 15.0
RETRY_SECONDS = 5.0
SUCCESS_GAP_SECONDS = 2.0


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def boot_id():
    path = Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.is_file() else None


def source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                     dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_reading(output):
    rows = list(csv.reader(io.StringIO(output.strip()), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError('expected exactly one uuid,name,memory.total CSV row')
    uuid, name, total = [field.strip() for field in rows[0]]
    return dict(uuid=uuid, name=name, memory_total_mib=int(total))


def wait_ready(gpu, max_wait_seconds=MAX_WAIT_SECONDS, *,
               runner=subprocess.run, now=time.monotonic, sleeper=time.sleep):
    """Use injected query/clock only for CPU fixtures; normal calls are read-only."""
    expected = IDENTITIES[gpu]
    command = ['nvidia-smi', '-i', gpu, '--query-gpu=uuid,name,memory.total',
               '--format=csv,noheader,nounits']
    started = now()
    deadline = started + max_wait_seconds
    consecutive = 0
    receipt = dict(status='waiting_for_driver', gpu=gpu, expected=expected,
                   command=command, started_utc=utc_now(), boot_id=boot_id(),
                   helper_pid=os.getpid(), source_sha256=source_sha(),
                   max_wait_seconds=max_wait_seconds,
                   retry_seconds=RETRY_SECONDS,
                   success_gap_seconds=SUCCESS_GAP_SECONDS,
                   required_consecutive_successes=2,
                   readiness_only=True, gpu_tasks_launched=0,
                   requires_original_queue_resource_clearance=True, attempts=[])

    def finish(status, reason):
        receipt.update(status=status, reason=reason, ended_utc=utc_now(),
                       elapsed_seconds=now() - started,
                       consecutive_successes=consecutive)
        return receipt

    while True:
        remaining = deadline - now()
        if remaining <= 0:
            return finish('failed', 'driver_readiness_timeout')
        attempt = dict(number=len(receipt['attempts']) + 1,
                       elapsed_seconds=now() - started, checked_utc=utc_now())
        try:
            response = runner(command, capture_output=True, text=True,
                              check=False,
                              timeout=min(QUERY_TIMEOUT_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            attempt.update(status='query_timeout')
        except OSError as error:
            attempt.update(status='command_launch_error', error=str(error))
            receipt['attempts'].append(attempt)
            return finish('failed', 'nvidia_smi_command_launch_error')
        else:
            attempt.update(returncode=response.returncode,
                           stdout=response.stdout.strip()[:2048],
                           stderr=response.stderr.strip()[:2048])
            if response.returncode == 9:
                attempt.update(status='driver_not_ready')
            elif response.returncode != 0:
                attempt.update(status='command_failed')
                receipt['attempts'].append(attempt)
                return finish('failed', 'unexpected_nvidia_smi_returncode')
            else:
                try:
                    reading = parse_reading(response.stdout)
                except (ValueError, TypeError) as error:
                    attempt.update(status='invalid_reading', error=str(error))
                    receipt['attempts'].append(attempt)
                    return finish('failed', 'invalid_nvidia_smi_reading')
                attempt['reading'] = reading
                if reading != expected:
                    attempt.update(status='identity_mismatch')
                    receipt['attempts'].append(attempt)
                    return finish('failed', 'physical_gpu_identity_mismatch')
                attempt.update(status='identity_verified')

        receipt['attempts'].append(attempt)
        # Even a successful query must return within the bounded admission window.
        if now() >= deadline:
            return finish('failed', 'driver_readiness_timeout')
        if attempt['status'] == 'identity_verified':
            consecutive += 1
            if consecutive == 2:
                return finish('ready', 'two_consecutive_identity_verified_readings')
            delay = SUCCESS_GAP_SECONDS
        else:
            consecutive = 0
            delay = RETRY_SECONDS
        sleeper(min(delay, max(0.0, deadline - now())))


def cpu_acceptance():
    """Exercise startup failure, identity rejection, and bounded timeout on CPU."""
    expected = IDENTITIES['0']
    success = ', '.join([expected['uuid'], expected['name'],
                         str(expected['memory_total_mib'])]) + '\n'
    wrong_uuid = success.replace(expected['uuid'], 'GPU-wrong-device')
    wrong_total = success.replace(str(expected['memory_total_mib']), '24576')
    wrong_name = success.replace(expected['name'], IDENTITIES['1']['name'])
    cases = [
        ('boot_exit9_then_success', [(9, ''), (0, success), (0, success)],
         600.0, 'ready', 'two_consecutive_identity_verified_readings', 7.0, 3),
        ('wrong_uuid_rejected', [(0, wrong_uuid)], 600.0, 'failed',
         'physical_gpu_identity_mismatch', 0.0, 1),
        ('wrong_memory_total_rejected', [(0, wrong_total)], 600.0, 'failed',
         'physical_gpu_identity_mismatch', 0.0, 1),
        ('wrong_name_rejected', [(0, wrong_name)], 600.0, 'failed',
         'physical_gpu_identity_mismatch', 0.0, 1),
        ('exit9_timeout_is_bounded', [(9, '')] * 3, 11.0, 'failed',
         'driver_readiness_timeout', 11.0, 3),
        ('intervening_failure_resets_pair',
         [(0, success), (9, ''), (0, success), (0, success)], 600.0,
         'ready', 'two_consecutive_identity_verified_readings', 9.0, 4),
        ('unexpected_error_is_not_hidden', [(2, '')], 600.0, 'failed',
         'unexpected_nvidia_smi_returncode', 0.0, 1),
        ('malformed_success_rejected', [(0, 'not-a-gpu\n')], 600.0, 'failed',
         'invalid_nvidia_smi_reading', 0.0, 1),
        ('query_timeout_then_success', [('timeout', ''), (0, success), (0, success)],
         600.0, 'ready', 'two_consecutive_identity_verified_readings', 22.0, 3),
        ('query_timeout_uses_remaining_budget', [('timeout', '')], 3.0, 'failed',
         'driver_readiness_timeout', 3.0, 1),
    ]
    checked = []
    for label, sequence, limit, status, reason, elapsed, count in cases:
        clock = [0.0]
        calls = []

        def fake_sleep(seconds):
            assert seconds >= 0
            clock[0] += seconds

        def fake_query(command, **kwargs):
            assert command[0] == 'nvidia-smi' and kwargs['check'] is False
            assert 0 < kwargs['timeout'] <= QUERY_TIMEOUT_SECONDS
            calls.append(list(command))
            code, output = sequence[len(calls) - 1]
            if code == 'timeout':
                clock[0] += kwargs['timeout']
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            return subprocess.CompletedProcess(command, code, stdout=output, stderr='')

        result = wait_ready('0', limit, runner=fake_query,
                            now=lambda: clock[0], sleeper=fake_sleep)
        assert result['status'] == status and result['reason'] == reason, result
        assert result['elapsed_seconds'] == elapsed and len(calls) == count, result
        if status == 'ready':
            assert result['consecutive_successes'] == 2
            assert (result['attempts'][-1]['elapsed_seconds'] -
                    result['attempts'][-2]['elapsed_seconds']) == SUCCESS_GAP_SECONDS
        checked.append(dict(case=label, status='passed', result=result))
    return dict(status='passed', acceptance='cpu_driver_admission_fixtures',
                source_sha256=source_sha(), nvidia_smi_queries_launched=0,
                gpu_tasks_launched=0, fixtures=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gpu', choices=sorted(IDENTITIES))
    mode.add_argument('--cpu-acceptance', action='store_true')
    parser.add_argument('--max-wait-seconds', type=float, default=MAX_WAIT_SECONDS)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.max_wait_seconds) or not (0 < args.max_wait_seconds <= MAX_WAIT_SECONDS):
        parser.error('--max-wait-seconds must be finite, positive, and at most 600')
    result = cpu_acceptance() if args.cpu_acceptance else wait_ready(args.gpu, args.max_wait_seconds)
    if args.receipt is not None:
        write(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['status'] in ['ready', 'passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
