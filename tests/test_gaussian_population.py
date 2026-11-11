"""Checkpoint population audits must inspect all PLY rows without CUDA/payload reads."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_gaussian_population import audit_model, ply_vertex_count


def header(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f'ply\nformat binary_little_endian 1.0\nelement vertex {count}\nproperty float x\nend_header\n'.encode())


class PopulationAuditTests(unittest.TestCase):
    def test_reads_only_header_and_reports_full_saved_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header(root/'input.ply', 10038407)
            header(root/'point_cloud/iteration_7000/point_cloud.ply', 10038407)
            header(root/'point_cloud/iteration_30000/point_cloud.ply', 8442)
            # Deliberately header-only: auditing must not allocate the declared payload.
            self.assertEqual(ply_vertex_count(root/'input.ply'), 10038407)
            (root/'cfg_args').write_text("Namespace(surface_mode='view_tangent', position_warmup_steps=7000)")
            report = audit_model(root)
            self.assertEqual([r['points'] for r in report['checkpoints']], [10038407, 8442])
            self.assertAlmostEqual(report['checkpoints'][-1]['fraction_of_input'], 8442/10038407)
            self.assertEqual(report['settings']['surface_mode'], 'view_tangent')
            self.assertTrue(any('less than 1%' in w for w in report['warnings']))

    def test_legacy_log_identifies_drop_but_does_not_invent_pruning_causes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [dict(iteration=10000, points_before=1000, added_children=40, pruned=10, points_after=1030),
                      dict(iteration=10100, points_before=1030, added_children=0, pruned=1000, points_after=30)]
            (root/'view_tangent_densification.jsonl').write_text('\n'.join(map(json.dumps, events)))
            report = audit_model(root)
            history = report['densification']
            self.assertEqual(history['largest_net_drops'][0]['iteration'], 10100)
            self.assertEqual(history['largest_net_drops'][0]['net_loss'], 1000)
            self.assertFalse(history['reasons_available_for_all_events'])
            self.assertNotIn('pruning_causes', history)
            self.assertTrue(any('not recorded' in w for w in report['warnings']))

    def test_disjoint_pruning_causes_distinguish_split_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = dict(iteration=10100, points_before=100, added_children=10, pruned=10, points_after=100,
                         replaced_split_parents=5, pruned_low_opacity=3, pruned_world_size=2, pruned_screen_size=0)
            path = root/'view_tangent_densification.jsonl'
            path.write_text(json.dumps(event)+'\n')
            report = audit_model(root)
            self.assertEqual(report['densification']['pruning_causes']['replaced_split_parents'], 5)
            self.assertEqual(report['densification']['largest_net_drops'], [])
            self.assertEqual(report['warnings'], [])
            event['points_after'] = 99
            path.write_text(json.dumps(event)+'\n{')
            broken = audit_model(root)
            self.assertTrue(any('accounting' in w for w in broken['warnings']))
            self.assertTrue(any('unreadable' in w for w in broken['warnings']))


if __name__ == '__main__':
    unittest.main()
