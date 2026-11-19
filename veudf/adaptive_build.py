"""Resume a frozen VE-UDF run with adaptive, memory-aware concurrency.

The scheduler lives outside the frozen package: changing scheduling does not
change the asset generation code or its recorded fingerprint.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import psutil

GIB = 1 << 30


def target_workers(available_gib, rss_gib, maximum, reserve=64, resume=72, per_worker=2, rss_limit=96):
    """A conservative cap, including currently running workers."""
    if available_gib < reserve or rss_gib >= rss_limit:
        return 0
    if available_gib < resume:
        return None  # Hold existing work but start nothing until headroom returns.
    return min(maximum, max(1, math.floor((available_gib - reserve) / per_worker)))


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def validate(base):
    from veudf.cli.prepare_remaining_run import digest
    from veudf.preprocess.pipeline import code_fingerprint, runtime_identity
    run = base / 'run'
    prep = json.loads((run / 'preparation.json').read_text())
    if digest(run / 'selected_manifest.jsonl') != prep['selected_manifest_sha256']:
        raise RuntimeError('Selected manifest changed')
    if digest(run / 'config.json') != prep['config_sha256']:
        raise RuntimeError('Configuration changed')
    for name, expected in json.loads((run / 'code_inventory.json').read_text()).items():
        if digest(run / 'code' / name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    if dict(code=code_fingerprint(), runtime=runtime_identity()) != prep['source_identity']:
        raise RuntimeError('Frozen code or runtime identity changed')
    return prep


class Limiter:
    def __init__(self, base, maximum, phase, reserve=64):
        from veudf.cli import build_dataset as driver
        from veudf.cli.run_guarded_build import memory_budget, tree_rss
        self.base, self.maximum, self.phase = base, maximum, phase
        self.reserve = reserve
        self.driver, self.memory_budget, self.tree_rss = driver, memory_budget, tree_rss
        self.condition = threading.Condition()
        self.running = 0
        self.allowed = 0
        self.closed = False
        self.cancelled = set()
        self.last_kill = 0.0
        self.available = 0.0
        self.rss = 0.0
        self.original_invoke = driver.invoke
        self.monitor = threading.Thread(target=self.watch, daemon=True)

    def start(self):
        self.monitor.start()
        self.driver.invoke = self.invoke

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.monitor.join(timeout=10)
        self.driver.invoke = self.original_invoke

    def emergency_kill_one(self):
        # Worker command ends in: ROOT SAMPLE_ID --record-stdin.
        with self.driver.LOCK:
            children = list(self.driver.CHILDREN)
        live = []
        for child in children:
            if child.poll() is not None:
                continue
            try:
                rss = psutil.Process(child.pid).memory_info().rss
                live.append((rss, child))
            except psutil.NoSuchProcess:
                pass
        if not live:
            return
        _, child = max(live, key=lambda pair: pair[0])
        sample = child.args[-2]
        with self.condition:
            self.cancelled.add(sample)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def watch(self):
        run = self.base / 'run'
        while not self.closed:
            try:
                _, available = self.memory_budget()
                rss = self.tree_rss(os.getpid())
                available, rss = available / GIB, rss / GIB
                cap = target_workers(available, rss, self.maximum, reserve=self.reserve, resume=self.reserve + 8)
                with self.condition:
                    self.available, self.rss = available, rss
                    self.allowed = 0 if cap is None else cap
                    active, allowed = self.running, self.allowed
                    self.condition.notify_all()
                state = ('waiting_for_memory' if not active else 'reducing_concurrency') if cap in (None, 0) else 'building'
                atomic(run / 'resource_status.json', dict(state=state, phase=self.phase,
                    supervisor_pid=os.getppid(), driver_pid=os.getpid(), workers_max=self.maximum,
                    workers_active=active, workers_allowed=allowed, available_gib=available,
                    job_rss_gib=rss, reserve_memory_gib=self.reserve, resume_memory_gib=self.reserve + 8,
                    emergency_memory_gib=self.reserve - 4, max_rss_gib=96, updated_unix=time.time()))
                if (available < self.reserve - 4 or rss > 96) and time.monotonic() - self.last_kill >= 5:
                    self.emergency_kill_one()
                    self.last_kill = time.monotonic()
                # A hard-stop safeguard applies even during serial publishing.
                if available < self.reserve - 8 or rss > 110:
                    self.terminate_children()
                    os._exit(75)
            except Exception as exc:
                # An unmonitored worker pool is not safe; outer supervisor retries.
                atomic(run / 'adaptive_error.json', dict(error=repr(exc), time=time.time()))
                self.terminate_children()
                os._exit(75)
            time.sleep(2)

    def terminate_children(self):
        with self.driver.LOCK:
            children = list(self.driver.CHILDREN)
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    def invoke(self, root, record, args):
        while not self.driver.STOP.is_set():
            with self.condition:
                while not self.closed and not self.driver.STOP.is_set() and self.running >= self.allowed:
                    self.condition.wait(timeout=2)
                if self.closed or self.driver.STOP.is_set():
                    break
                self.running += 1
            try:
                result = self.original_invoke(root, record, args)
            finally:
                with self.condition:
                    self.running -= 1
                    self.condition.notify_all()
            with self.condition:
                interrupted = record['sample_id'] in self.cancelled
                self.cancelled.discard(record['sample_id'])
            if not (interrupted and result.get('code') == 'worker_crash'):
                return result
        return dict(status='failed', code='worker_crash', sample_id=record['sample_id'], split=record['split'])


def worker(base, phase, maximum, reserve=64):
    from veudf.cli import build_dataset as driver
    from veudf.cli.run_guarded_build import cpu_budget
    prep = validate(base)
    workers = min(maximum, math.floor(cpu_budget() - 16))
    if workers < 1:
        raise RuntimeError('CPU quota cannot preserve the requested reserve')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:workers])
    os.nice(10)
    limit = min(500, prep['selected']) if phase == 'pilot' else prep['selected']
    limiter = Limiter(base, workers, phase, reserve)
    limiter.start()
    try:
        return driver.main(['build', '--root', str(base/'data'), '--manifest', str(base/'run/selected_manifest.jsonl'),
                            '--config', str(base/'run/config.json'), '--limit', str(limit),
                            '--workers', str(workers), '--threads', '1', '--timeout', '7200'])
    finally:
        limiter.close()


def pilot_audit(base, expected):
    report = json.loads((base/'data/report.json').read_text())
    status = json.loads((base/'data/status.json').read_text())
    index = json.loads((base/'data/index.json').read_text())
    problems = []
    if status['state'] != 'finished' or report['candidates'] != expected or report['attempted'] != expected:
        problems.append('pilot_not_complete')
    if report['published'] < 1 or report['published'] != len(index['assets']):
        problems.append('no_or_invalid_published_assets')
    critical = {'worker_crash', 'timeout', 'source_hash_mismatch', 'output_exists', 'unknown', 'exception', 'quality_failed', 'distance_solver_failed', 'sdf_sign_failed'}
    systemic = {k:v for k,v in report['failures'].items() if k in critical}
    if systemic: problems.append('systemic_failure_codes')
    samples = index['assets'][:32]
    mesh, refs, unresolved = [], [], {'edge':0, 'vertex':0}
    for a in samples:
        m = json.loads((base/'data/assets'/a['asset_id']/'meta.json').read_text())
        mesh.append(float(m['quality']['mesh_vs_cad']['max']))
        refs.extend(float(x) for x in m['quality']['reference_max_error'])
        for k in unresolved: unresolved[k] += len(m['reachability'][k]['unresolved'])
    if not all(math.isfinite(x) for x in mesh + refs): problems.append('nonfinite_quality')
    result = dict(report={k:v for k,v in report.items() if k != 'dedup_rejected'}, sample_count=len(samples),
                  mesh_vs_cad_max=max(mesh, default=None), reference_max_error=max(refs, default=None),
                  unresolved_instances=unresolved, systemic_failures=systemic, problems=problems,
                  note='No numerical mesh-vs-CAD acceptance threshold is configured; the pilot is not guaranteed representative.')
    atomic(base/'run/pilot_audit.json', result)
    if problems: raise RuntimeError('Pilot audit requires review: ' + ', '.join(problems))


def supervise(base, maximum, reserve=64):
    from veudf.cli.run_guarded_build import memory_budget
    prep = validate(base)
    run = base/'run'
    lock = (run/'resource_guard.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another resource guard is already running')
    stopped = False
    child = None
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = dict(os.environ, PYTHONPATH=str(run/'code'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
    phases = ['pilot', 'full']
    for phase in phases:
        while not stopped:
            # Leave at least 8 GiB of headroom above the reservation before starting.
            _, available = memory_budget()
            atomic(run/'adaptive_supervisor_status.json', dict(state='waiting_for_memory' if available/GIB < reserve + 8 else 'starting', phase=phase, supervisor_pid=os.getpid(), available_gib=available/GIB, reserve_memory_gib=reserve, updated_unix=time.time()))
            if available/GIB < reserve + 8:
                time.sleep(5)
                continue
            with (run/'adaptive_build.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run-dir', str(base), '--max-workers', str(maximum), '--reserve-memory-gib', str(reserve), '--worker-phase', phase], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic(run/'adaptive_supervisor_status.json', dict(state='running', phase=phase, supervisor_pid=os.getpid(), driver_pid=child.pid, updated_unix=time.time()))
                while child.poll() is None:
                    time.sleep(2)
                rc = child.returncode
                child = None
            if stopped: break
            if rc == 75:
                # Extreme pressure or monitoring failure: leave results intact and retry.
                time.sleep(5)
                continue
            if rc != 0: raise RuntimeError(f'{phase} driver exited {rc}; see adaptive_build.log')
            if phase == 'pilot': pilot_audit(base, min(500, prep['selected']))
            break
        if stopped:
            atomic(run/'adaptive_supervisor_status.json', dict(state='cancelled', phase=phase, updated_unix=time.time()))
            return 130
    report = json.loads((base/'data/report.json').read_text())
    status = json.loads((base/'data/status.json').read_text())
    if status['state'] != 'finished' or report['candidates'] != prep['selected'] or report['attempted'] != prep['selected']:
        raise RuntimeError('Final counts do not match selected candidates')
    atomic(run/'completion_audit.json', dict(report={k:v for k,v in report.items() if k != 'dedup_rejected'}, state=status['state'], note='Cross-root geometric deduplication is not performed.'))
    atomic(run/'adaptive_supervisor_status.json', dict(state='finished', supervisor_pid=os.getpid(), updated_unix=time.time()))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--max-workers', type=int, default=48)
    p.add_argument('--reserve-memory-gib', type=float, default=64)
    p.add_argument('--worker-phase', choices=('pilot', 'full'))
    a = p.parse_args()
    base = Path(a.run_dir).resolve()
    if a.max_workers < 1: p.error('max-workers must be positive')
    if not math.isfinite(a.reserve_memory_gib) or a.reserve_memory_gib <= 8: p.error('reserve-memory-gib must be finite and greater than 8')
    if a.worker_phase: return worker(base, a.worker_phase, a.max_workers, a.reserve_memory_gib)
    try:
        return supervise(base, a.max_workers, a.reserve_memory_gib)
    except Exception as exc:
        atomic(base/'run/adaptive_supervisor_status.json', dict(state='stopped', error=repr(exc), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
