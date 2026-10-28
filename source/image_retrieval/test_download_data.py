import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import download_data


class DownloadTests(unittest.TestCase):
    def bundle(self):
        content=b'fixture-artifact'
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            z.writestr('data/test.jpg',b'image')
            z.writestr('artifacts/images.parquet',content)
            z.writestr('artifacts/run_manifest.json',json.dumps({'outputs':{'images.parquet':hashlib.sha256(content).hexdigest()}}))
        return output.getvalue()

    def test_downloads_wrapped_bundle_checks_hash_and_installs_paths(self):
        bundle=self.bundle()
        package=io.BytesIO()
        with zipfile.ZipFile(package,'w') as z: z.writestr('bundle.zip',bundle)
        class Response:
            status_code=200
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def raise_for_status(self): pass
            def iter_content(self,*args): yield package.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'.env.example').write_text('KAGGLE_API_TOKEN=\n')
            (root/'public_data.json').write_text(json.dumps({'dataset':'owner/fixture','download_url':'https://github.com/owner/repo/releases/download/data-v1/bundle.zip','sha256':hashlib.sha256(bundle).hexdigest(),'images':1,'objects':0}))
            with patch.object(download_data,'ROOT',root),patch.object(download_data,'HERE',root),patch('download_data.requests.get',return_value=Response()) as download,patch.dict(os.environ,{'IMAGE_DATA_DIR':str(root/'data'),'IMAGE_ARTIFACT_DIR':str(root/'artifacts'),'KAGGLE_API_TOKEN':'fixture-secret-not-for-github'}):
                download_data.main()
            self.assertEqual(download.call_args.kwargs['headers'],{})
            self.assertTrue((root/'.env').exists())
            self.assertEqual((root/'data/test.jpg').read_bytes(),b'image')
            self.assertEqual((root/'artifacts/images.parquet').read_bytes(),b'fixture-artifact')

    def test_rejects_traversal_before_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('data/../../escape.txt','unsafe')
            with self.assertRaises(ValueError): download_data.extract_bundle(archive,root/'staging')
            self.assertFalse((root/'escape.txt').exists())

    def test_checksum_survives_kaggle_repacking_and_rejects_changed_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=root/'original.zip';original.write_bytes(self.bundle())
            expected=download_data.extract_bundle(original,root/'first')
            with zipfile.ZipFile(original) as source:
                entries=[(name,source.read(name)) for name in source.namelist()]
            repacked=root/'repacked.zip'
            with zipfile.ZipFile(repacked,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for name,content in reversed(entries):z.writestr(name,content)
            self.assertEqual(download_data.extract_bundle(repacked,root/'second',expected,3),expected)
            with self.assertRaises(ValueError):download_data.extract_bundle(repacked,root/'third','0'*64,3)


if __name__=='__main__': unittest.main()
