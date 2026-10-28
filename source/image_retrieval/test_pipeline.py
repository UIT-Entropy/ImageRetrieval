"""CPU contract tests for input identity, foreground colors and resume safety."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from common import (atomic_json, build_manifest, color_distribution, digest_file,
                    read_embedding_shard, smoke_indices, validate_vectors)
from pipeline import read_object_checkpoint, write_parquet, object_schema
from manage import generate_notebook, scan_package
import manage


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        batch = self.root / 'Batch1' / 'Batch1'
        batch.mkdir(parents=True)
        rows = []
        for hour in [8, 9]:
            Image.new('RGB', (8, 6), 'red').save(batch / f'{hour:02}_10.jpg')
            rows.append({'imageID': f'day1_Kitchen_{hour:02}_10', 'owner': 'Kitchen', 'day': 'day1',
                         'hour': hour, 'frame': 10, 'class': '["cup"]', 'class_count': '{"cup": 1}',
                         'description': 'Original caption retained as source metadata.'})
        with (self.root / 'metadata_sample.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def test_manifest_retains_metadata_and_id_mapping(self):
        self.fixture()
        rows = build_manifest(self.root, expected=2)
        self.assertEqual(rows[0]['imageID'], 'day1_Kitchen_08_10')
        self.assertEqual(rows[0]['path'], 'images/day1_Kitchen_08_10.jpg')
        self.assertEqual(rows[0]['class_count'], '{"cup": 1}')
        self.assertNotIn('timestamp_ms', rows[0])

    def test_missing_or_ambiguous_image_rejected(self):
        self.fixture()
        duplicate = self.root / 'Batch2'
        duplicate.mkdir()
        Image.new('RGB', (2, 2)).save(duplicate / '08_10.jpg')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            build_manifest(self.root, 2)

    def test_color_uses_foreground_and_keeps_multiple_colors(self):
        image = np.array([[[255,0,0], [255,255,255]], [[0,0,255], [255,255,255]]], dtype=np.uint8)
        mask = np.array([[True,False],[True,False]])
        colors = color_distribution(image, mask)
        self.assertEqual(colors, {'blue': .5, 'red': .5})
        with self.assertRaises(ValueError):
            color_distribution(image, np.zeros((2,2), bool))

    def test_embedding_checkpoint_order_and_corruption(self):
        path = self.root / 'vectors.npz'
        np.savez(path, ids=np.array(['a', 'b']), vectors=np.eye(2, dtype='float32'))
        np.testing.assert_equal(read_embedding_shard(path, ['a','b'], 2), np.eye(2))
        with self.assertRaisesRegex(ValueError, 'IDs mismatch'):
            read_embedding_shard(path, ['b','a'], 2)
        with self.assertRaises(ValueError):
            validate_vectors(np.array([[float('nan'),0]]), 1, 2)
        with self.assertRaises(ValueError):
            validate_vectors(np.zeros((1,2)), 1, 2)

    def test_object_checkpoint_validates_mask_and_image(self):
        image = self.root / 'mask.png'
        Image.new('L', (4, 4), 255).save(image)
        record = {'imageID':'a', 'image_sha256':'hash', 'objects':[
            {'mask_path':'mask.png', 'mask_sha256':digest_file(image)}]}
        path = self.root / 'checkpoint.json'
        atomic_json(path, record)
        row = {'imageID':'a', 'sha256':'hash'}
        self.assertEqual(len(read_object_checkpoint(path, row, self.root)), 1)
        Image.new('L', (4,4), 0).save(image)
        with self.assertRaisesRegex(ValueError, 'corrupt mask'):
            read_object_checkpoint(path, row, self.root)

    def test_smoke_covers_all_hours_without_duplicates(self):
        rows = [{'day':'day1','source':'Kitchen','hour':hour} for hour in range(8,21) for _ in range(10)]
        chosen = smoke_indices(rows, 20)
        self.assertEqual(len(set(chosen)), 20)
        self.assertEqual({rows[i]['hour'] for i in chosen}, set(range(8,21)))

    def test_notebook_compiles_and_contains_no_credentials(self):
        generate_notebook(self.root, {'dataset':'test/input', 'kernel':'test/preprocess'})
        notebook = json.loads((self.root/'preprocess.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<notebook-cell>', 'exec')
        self.assertTrue(scan_package(self.root))
        (self.root/'secret.json').write_text('{"token":"KGAT_fake"}')
        with self.assertRaisesRegex(ValueError, 'credential'):
            scan_package(self.root)

    def test_empty_object_output_has_stable_schema(self):
        import pyarrow.parquet as pq
        path = self.root/'objects.parquet'
        write_parquet(path, [], object_schema())
        table = pq.read_table(path)
        self.assertEqual(table.num_rows, 0)
        self.assertIn('colors_json', table.column_names)

    def test_remote_privacy_handles_sdk_info_wrapper(self):
        class FakeApi:
            private = True

            def dataset_metadata(api, dataset, folder):
                atomic_json(Path(folder)/'dataset-metadata.json', {'info': {'isPrivate': api.private}})

            def kernels_pull(api, kernel, folder, **kwargs):
                atomic_json(Path(folder)/'kernel-metadata.json', {'is_private': True})

        api = FakeApi()
        with patch.object(manage, 'WORK', self.root), \
             patch.object(manage, 'state', return_value={'dataset':'test/data','kernel':'test/run','kernel_pushed':True}), \
             patch.object(manage, 'authenticate', return_value=(api,'test')):
            manage.verify_private(None)
            api.private = False
            with self.assertRaisesRegex(RuntimeError, 'privacy'):
                manage.verify_private(None)


if __name__ == '__main__':
    unittest.main()
