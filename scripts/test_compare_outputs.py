"""Regression checks for portability and detection of meaningful mismatches."""
import tempfile
import unittest
from pathlib import Path
from compare_outputs import compare_files, compare_values

class ComparisonTests(unittest.TestCase):
    def test_newlines(self):
        fixtures = {'.json': '{"n": 185, "value": 0.9844}\n',
                    '.jsonl': '{"id": 1, "value": 0.9844}\n{"id": 2, "value": 0.0}\n',
                    '.csv': 'group_id,value\n001,0.9844\n002,0.0\n'}
        with tempfile.TemporaryDirectory() as folder:
            for suffix, text in fixtures.items():
                left, right = Path(folder)/('a'+suffix), Path(folder)/('b'+suffix)
                left.write_bytes(text.replace('\n', '\r\n').encode())
                right.write_bytes(text.encode())
                result = compare_files(left, right)
                self.assertTrue(result['match'])
                self.assertFalse(result['byte_identical'])

    def test_roundoff_and_zero(self):
        compare_values([0.9844, 0.0], [0.9844 + 1e-12, 1e-14])
        for pair in [(0.0, 1e-8), (0.9844, 0.985), (1e-10, 2e-10)]:
            with self.assertRaises(AssertionError):
                compare_values(*pair)

    def test_exact_data(self):
        for pair in [(185, 186), (185, 185.0), (True, 1), ('001', '1'),
                     ({'sha256':'abc'}, {'sha256':'def'}), ([1], [1,2]),
                     ({'a':1}, {'b':1}), (None, 0)]:
            with self.assertRaises(AssertionError):
                compare_values(*pair)

    def test_nonfinite(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(AssertionError):
                compare_values(value, value)

    def test_csv_record_id(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder)/'a.csv', Path(folder)/'b.csv'
            left.write_text('group_id,value\n001,0.5\n')
            right.write_text('group_id,value\n1,0.5\n')
            with self.assertRaises(AssertionError):
                compare_files(left, right)

    def test_csv_nonfinite(self):
        with tempfile.TemporaryDirectory() as folder:
            left = Path(folder)/'a.csv'
            left.write_text('value\nnan\n')
            with self.assertRaises(AssertionError):
                compare_files(left, left)

if __name__ == '__main__':
    unittest.main()
