"""Run one explicitly authorized full-state recovery, then finish the matrix."""
from common import *
import argparse, fcntl, traceback
import run_matrix as matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    lock = (RUN / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        authorization = read(args.authorization)
        assert authorization['status'] == 'authorized'
        scene, mode = authorization['scene'], authorization['mode']
        assert scene in config()['scenes'] and mode in config()['modes']
        checkpoint = authorization['checkpoint']
        assert sha(checkpoint['path']) == checkpoint['sha256']
        output = scene_dir(scene) / 'runs' / mode
        assert output.exists() and not (output / 'run.json').exists()
        assert not (output / 'resume_receipt.json').exists()
        # Observability only: do not change numerical execution or training code.
        os.environ['PYTHONFAULTHANDLER'] = '1'
        save_json(Path(args.authorization).parent / 'recovery_controller.json', dict(
            pid=os.getpid(), time_unix=time.time(), authorization=identity(args.authorization),
            checkpoint=checkpoint, fault_handler=True, optimizer_recovery_limit=1))
        matrix.job(scene + '_' + mode + '_resume1', 'train_scene.py', [
            '--scene', scene, '--mode', mode, '--resume', checkpoint['path'],
            '--external-reason', authorization['reason']])
        matrix.train()
        matrix.evaluate()
        matrix.report()
    except BaseException:
        previous = read(RUN / 'pipeline.json')
        save_json(RUN / 'pipeline.json', dict(status='failed',
            requested_phase='explicitly_authorized_recovery_to_report',
            last_phase=previous.get('phase'), pid=os.getpid(), time_unix=time.time(),
            traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
