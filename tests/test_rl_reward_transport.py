"""Real pipe regressions for the reward worker's complete-response deadline."""
import json
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from llava.train.rl.pixel_reward import QwenPixelReward


class RewardTransportTests(unittest.TestCase):
    def start_worker(self, code, timeout=5):
        popen = subprocess.Popen
        with patch('llava.train.rl.pixel_reward.subprocess.Popen',
                   side_effect=lambda *a, **kw: popen([sys.executable, '-S', '-u', '-c', code], **kw)):
            judge = QwenPixelReward(sys.executable, 'unused', 'cpu', timeout=timeout)
        self.addCleanup(judge.close)
        return judge

    def test_silent_partial_and_dripping_responses_obey_one_deadline(self):
        for output in ('time.sleep(2)',
                       'sys.stdout.write("{"); sys.stdout.flush(); time.sleep(2)',
                       'sys.stdout.write("{"); sys.stdout.flush()\n'
                       'for _ in range(100):\n'
                       ' sys.stdout.write(" "); sys.stdout.flush(); time.sleep(.03)'):
            with self.subTest(output=output):
                judge = self.start_worker('import sys,time\nprint(\'{"ready":true}\',flush=True)\n'
                                         'sys.stdin.readline()\n' + output)
                judge.timeout = .15
                judge.process.stdin.write('go\n')
                judge.process.stdin.flush()
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    judge._read()
                self.assertLess(time.monotonic() - start, 1.0)
                self.assertIsNotNone(judge.process.poll())

    def test_partial_startup_times_out_and_reaps_worker(self):
        popen = subprocess.Popen
        processes = []
        def launch(*a, **kw):
            process = popen([sys.executable, '-S', '-u', '-c',
                'import sys,time; sys.stdout.write("{"); sys.stdout.flush(); time.sleep(5)'], **kw)
            processes.append(process)
            return process
        with patch('llava.train.rl.pixel_reward.subprocess.Popen', side_effect=launch):
            with self.assertRaises(TimeoutError):
                QwenPixelReward(sys.executable, 'unused', 'cpu', timeout=.2)
        self.assertIsNotNone(processes[0].poll())

    def test_fragmented_utf8_and_multiple_buffered_lines(self):
        payload = json.dumps({'message': 'caf\u00e9'}, ensure_ascii=False).encode() + b'\n'
        judge = self.start_worker(
            'import sys,time\nprint(\'{"ready":true}\',flush=True)\nsys.stdin.readline()\n'
            f'payload={payload!r}\n'
            'for byte in payload:\n'
            ' sys.stdout.buffer.write(bytes([byte])); sys.stdout.flush(); time.sleep(.002)\n'
            'sys.stdout.write(\'{"a":1}\\n{"b":2}\\n\'); sys.stdout.flush()\n'
            'time.sleep(5)')
        judge.process.stdin.write('go\n')
        judge.process.stdin.flush()
        self.assertEqual(judge._read(), {'message': 'caf\u00e9'})
        self.assertEqual(judge._read(), {'a': 1})
        self.assertEqual(judge._read(), {'b': 2})

    def test_eof_before_newline_and_malformed_complete_line(self):
        for output, error in [(b'', RuntimeError), (b'{', RuntimeError),
                              (b'{"ready":true}', RuntimeError), (b'{invalid}\n', json.JSONDecodeError)]:
            with self.subTest(output=output):
                judge = self.start_worker('import sys\nprint(\'{"ready":true}\',flush=True)\n'
                    f'sys.stdin.readline()\nsys.stdout.buffer.write({output!r}); sys.stdout.flush()')
                judge.process.stdin.write('go\n')
                judge.process.stdin.flush()
                with self.assertRaises(error):
                    judge._read()


if __name__ == '__main__':
    unittest.main()
