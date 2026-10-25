"""The synced input folder: build, manifest, verify and publish."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/clusters'))
import shared_inputs as si  # noqa: E402


def fixture(tmp):
    src = Path(tmp)/'src'
    model = src/'sft'
    model.mkdir(parents=True)
    (model/'config.json').write_text('{}')
    (model/'model-00001-of-00001.safetensors').write_bytes(b'w' * 100)
    (model/'rng_state_0.pth').write_bytes(b'r' * 10)
    (src/'ar.pth').write_bytes(b'a' * 50)
    data = src/'data'
    (data/'sub').mkdir(parents=True)
    (data/'sub/train.jsonl').write_text('{"p": 1}\n')
    sources = Path(tmp)/'sources.txt'
    sources.write_text(f"""# comment
copy-model {model} models/sft
link       {src/'ar.pth'} models/tar/ar.pth   # same filesystem
copy       {data} data
""")
    return src, sources, Path(tmp)/'shared'


class SharedInputsTest(unittest.TestCase):
    def test_build_places_inputs_writes_manifest_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, sources, dest = fixture(tmp)
            self.assertEqual(si.main(['build', '--dest', str(dest), '--sources', str(sources)]), 0)
            self.assertEqual(sorted(p.name for p in (dest/'models/sft').iterdir()),
                             ['config.json', 'model-00001-of-00001.safetensors'])
            self.assertEqual(os.stat(dest/'models/tar/ar.pth').st_ino, os.stat(src/'ar.pth').st_ino)
            files = json.loads((dest/si.MANIFEST).read_text())['files']
            self.assertEqual(set(files), {'models/sft/config.json', 'models/sft/model-00001-of-00001.safetensors',
                                          'models/tar/ar.pth', 'data/sub/train.jsonl', si.PROVENANCE})
            provenance = json.loads((dest/si.PROVENANCE).read_text())
            self.assertEqual(provenance['models/tar/ar.pth']['source'], str(src/'ar.pth'))
            self.assertEqual(si.verify(dest, full=True), [])

    def test_rebuild_keeps_complete_inputs_and_refuses_changed_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, sources, dest = fixture(tmp)
            si.main(['build', '--dest', str(dest), '--sources', str(sources)])
            self.assertEqual(si.main(['build', '--dest', str(dest), '--sources', str(sources)]), 0)
            (src/'data/sub/train.jsonl').write_text('{"p": 2, "changed": true}\n')
            with self.assertRaises(FileExistsError):
                si.main(['build', '--dest', str(dest), '--sources', str(sources)])

    def test_verify_reports_truncation_missing_and_stray_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sources, dest = fixture(tmp)
            si.main(['build', '--dest', str(dest), '--sources', str(sources)])
            (dest/'data/sub/train.jsonl').write_bytes(b'')           # e.g. a scratch incident
            (dest/'models/sft/config.json').unlink()                   # e.g. purged
            (dest/'stray.txt').write_text('x')
            problems = si.verify(dest)
            self.assertTrue(any(p.startswith('size 0') for p in problems))
            self.assertIn('missing: models/sft/config.json', problems)
            self.assertIn('not in manifest: stray.txt', problems)
            self.assertEqual(si.main(['verify', '--dest', str(dest)]), 1)

    def test_publish_adds_one_input_and_touches_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sources, dest = fixture(tmp)
            si.main(['build', '--dest', str(dest), '--sources', str(sources)])
            old = time.time() - 3600
            os.utime(dest, (old, old))
            ck = Path(tmp)/'checkpoint-100'
            ck.mkdir()
            (ck/'adapter_model.safetensors').write_bytes(b'l' * 20)
            (ck/'adapter_config.json').write_text('{}')
            (ck/'optimizer.pt').write_bytes(b'o' * 30)
            self.assertEqual(si.main(['publish', '--dest', str(dest), str(ck), 'checkpoints/run/checkpoint-100']), 0)
            self.assertTrue((dest/'checkpoints/run/checkpoint-100/optimizer.pt').is_file())  # full copy
            self.assertGreater(dest.stat().st_mtime, old + 60)
            self.assertEqual(si.verify(dest, full=True), [])

    def test_sources_must_stay_inside_the_folder(self):
        for line in ('copy /x ../escape', 'copy /x /abs', 'move /x y', 'copy /x'):
            with self.assertRaises(ValueError):
                si.parse_sources(line)

    def test_model_mode_requires_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'m').mkdir()
            (Path(tmp)/'m/config.json').write_text('{}')
            with self.assertRaises(ValueError):
                si.place('copy-model', Path(tmp)/'m', Path(tmp)/'out')


if __name__ == '__main__':
    unittest.main()
