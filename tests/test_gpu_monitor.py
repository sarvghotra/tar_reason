import json
import fcntl
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.monitor_gpu_test import check_once


class MonitorTests(unittest.TestCase):
    def test_overlapping_cron_invocation_does_not_query_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "results/logs/gpu-test-123"
            directory.mkdir(parents=True)
            with (directory / "monitor.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                def runner(*args, **kwargs):
                    self.fail("An overlapping cron invocation must exit without work.")
                check_once("123", root, runner)

    def test_wait_then_run_once_even_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ["PENDING"]
            executions = []
            def runner(command, **kwargs):
                if command[0] == "squeue":
                    return SimpleNamespace(returncode=0, stdout=state[0], stderr="")
                executions.append(command)
                self.assertIn("--jobid=123", command)
                self.assertIn("--overlap", command)
                return SimpleNamespace(returncode=1)
            check_once("123", root, runner)
            directory = root / "results/logs/gpu-test-123"
            self.assertEqual(json.loads((directory / "status.json").read_text())["state"], "waiting")
            self.assertFalse(executions)
            state[0] = "RUNNING"
            check_once("123", root, runner)
            check_once("123", root, runner)
            self.assertEqual(len(executions), 1)
            self.assertEqual(json.loads((directory / "status.json").read_text())["state"], "failed")

    def test_missing_allocation_does_not_start_another_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def runner(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            root = Path(tmp)
            check_once("123", root, runner)
            check_once("123", root, runner)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "squeue")

    def test_transient_query_error_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            def runner(command, **kwargs):
                return SimpleNamespace(returncode=1, stdout="", stderr="temporary failure")
            root = Path(tmp)
            check_once("123", root, runner)
            directory = root / "results/logs/gpu-test-123"
            self.assertFalse((directory / "attempted").exists())
            self.assertFalse((directory / "finished").exists())


if __name__ == "__main__":
    unittest.main()
