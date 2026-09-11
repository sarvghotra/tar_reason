import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from scripts.clusters.setup_links import configure


class LinkTests(unittest.TestCase):
    def fixture(self, tmp):
        root=Path(tmp)/'checkout'; root.mkdir()
        profile=root/'scripts/clusters/fir'; profile.mkdir(parents=True)
        (profile/'profile.sh').write_text('')
        targets={}
        for name in ('data','models','sft_model','reward_model','results'):
            target=Path(tmp)/('external-'+name);target.mkdir();targets[name]=target
        return root,targets

    def test_idempotent_and_targets_untouched(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root,targets=self.fixture(tmp)
            (targets['data']/'keep').write_text('original')
            configure(root,'fir',targets)
            configure(root,'fir',targets)
            self.assertEqual((root/'.cluster').read_text(),'fir\n')
            self.assertEqual((root/'data/keep').read_text(),'original')
            self.assertTrue(all((root/k).is_symlink() for k in targets))

    def test_preflight_prevents_partial_changes_and_real_directory_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,targets=self.fixture(tmp)
            (root/'results').mkdir()
            with self.assertRaisesRegex(ValueError,'real file/directory'):
                configure(root,'fir',targets,replace=True)
            self.assertFalse((root/'data').exists())
            self.assertFalse((root/'.cluster').exists())

    def test_retarget_requires_flag_and_preserves_old_target(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root,targets=self.fixture(tmp)
            configure(root,'fir',targets)
            old=targets['data']; (old/'keep').touch()
            targets['data']=Path(tmp)/'new-data';targets['data'].mkdir()
            with self.assertRaisesRegex(ValueError,'replace-links'):
                configure(root,'fir',targets)
            configure(root,'fir',targets,replace=True)
            self.assertEqual((root/'data').resolve(),targets['data'])
            self.assertTrue((old/'keep').exists())

    def test_missing_target_does_not_create_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,targets=self.fixture(tmp)
            targets['results']=Path(tmp)/'missing'
            with self.assertRaisesRegex(ValueError,'missing'):
                configure(root,'fir',targets)
            self.assertFalse((root/'data').exists())

if __name__=='__main__': unittest.main()
