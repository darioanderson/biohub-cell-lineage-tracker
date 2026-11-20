"""Test subset budgeting and embryo isolation without network access."""
import unittest
from download_biohub_subset import select


def sample(name):
    prefix = 'train/'+name
    return [{'name': prefix+'.zarr/0/zarr.json', 'bytes': 100},
            {'name': prefix+'.zarr/zarr.json', 'bytes': 100},
            {'name': prefix+'.zarr/0/c/0/0/0/0', 'bytes': 1000},
            {'name': prefix+'.zarr/0/c/1/0/0/0', 'bytes': 1000},
            {'name': prefix+'.geff/zarr.json', 'bytes': 100},
            {'name': prefix+'.geff/nodes/ids/c/0', 'bytes': 100}]


class SubsetChecks(unittest.TestCase):
    def test_different_embryos_and_only_selected_frames(self):
        p = select(sample('aaa_01')+sample('aaa_02')+sample('bbb_01'), [0], 3000)
        self.assertEqual([s['embryo'] for s in p['samples']], ['aaa', 'bbb'])
        self.assertEqual(p['planned_bytes'], 2800)
        self.assertTrue(all('/c/1/' not in f['name'] for f in p['files']))

    def test_cap_is_enforced(self):
        with self.assertRaises(ValueError):
            select(sample('aaa_01')+sample('bbb_01'), [0], 2000)

    def test_missing_frame_is_not_zero_image(self):
        with self.assertRaises(ValueError):
            select(sample('aaa_01')+sample('bbb_01'), [3], 9000)


if __name__ == '__main__':
    unittest.main()
