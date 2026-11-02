import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('adaptive_build', Path(__file__).parents[1] / 'adaptive_build.py')
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


def test_worker_budget_and_hysteresis():
    assert adaptive.target_workers(63, 10, 47) == 0
    assert adaptive.target_workers(70, 10, 47) is None
    assert adaptive.target_workers(72, 10, 47) == 4
    assert adaptive.target_workers(160, 10, 47) == 47
    assert adaptive.target_workers(100, 96, 47) == 0
    assert adaptive.target_workers(91, 10, 47, reserve=32, resume=40) == 29
    assert adaptive.target_workers(31, 10, 47, reserve=32, resume=40) == 0
    assert adaptive.target_workers(35, 10, 47, reserve=32, resume=40) is None


def test_waiting_does_not_invoke_worker_and_releases_on_capacity():
    called = []
    controller = object.__new__(adaptive.Limiter)
    controller.condition = threading.Condition()
    controller.running = 0
    controller.allowed = 0
    controller.closed = False
    controller.cancelled = set()
    controller.driver = SimpleNamespace(STOP=threading.Event())
    controller.original_invoke = lambda root, record, args: called.append(record['sample_id']) or {'status': 'passed'}
    record = {'sample_id': 'one', 'split': 'train'}
    result = []
    thread = threading.Thread(target=lambda: result.append(controller.invoke(None, record, None)))
    thread.start()
    time.sleep(0.05)
    assert not called
    with controller.condition:
        controller.allowed = 1
        controller.condition.notify_all()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert called == ['one']
    assert result == [{'status': 'passed'}]
    assert controller.running == 0


def test_emergency_interrupted_worker_is_retried():
    controller = object.__new__(adaptive.Limiter)
    controller.condition = threading.Condition()
    controller.running = 0
    controller.allowed = 1
    controller.closed = False
    controller.cancelled = set()
    controller.driver = SimpleNamespace(STOP=threading.Event())
    calls = []
    def invoke(root, record, args):
        calls.append(1)
        if len(calls) == 1:
            controller.cancelled.add(record['sample_id'])
            return {'status': 'failed', 'code': 'worker_crash'}
        return {'status': 'passed'}
    controller.original_invoke = invoke
    assert controller.invoke(None, {'sample_id': 'one', 'split': 'train'}, None)['status'] == 'passed'
    assert len(calls) == 2
