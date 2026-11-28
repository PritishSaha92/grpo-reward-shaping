"""Compare regenerated data without weakening the separate input-hash checks.

Only finite floating-point values receive tolerance. Structure, integer counts,
identifiers, strings, booleans and nulls must match exactly. CSV cells containing
decimal/exponent notation are numerical unless the column identifies a record.
"""
import csv
import io
import json
import math
import re
from pathlib import Path

RTOL = 1e-9
ATOL = 1e-12  # absolute allowance for roundoff at zero, not missing observations
DECIMAL = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$")
IDENTIFIER = re.compile(r"(?:^|_)(?:id|ids|index|idx|seed|hash|sha256|count|counts)(?:$|_)")

def compare_values(expected, actual, path='$', stats=None):
    if stats is None:
        stats = {'float_values': 0, 'max_absolute_error': 0.0, 'max_relative_error': 0.0}
    if type(expected) is not type(actual):
        raise AssertionError(f'{path}: type differs ({type(expected).__name__}, {type(actual).__name__})')
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in expected:
            compare_values(expected[key], actual[key], f'{path}.{key}', stats)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f'{path}: length differs')
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f'{path}[{i}]', stats)
    elif isinstance(expected, float):
        if not (math.isfinite(expected) and math.isfinite(actual)):
            raise AssertionError(f'{path}: non-finite numerical value')
        error = abs(expected - actual)
        scale = max(abs(expected), abs(actual))
        stats['float_values'] += 1
        stats['max_absolute_error'] = max(stats['max_absolute_error'], error)
        stats['max_relative_error'] = max(stats['max_relative_error'], error / scale if scale else 0.0)
        if not math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f'{path}: {expected!r} != {actual!r} within rtol={RTOL}, atol={ATOL}')
    elif expected != actual:
        raise AssertionError(f'{path}: exact value differs ({expected!r}, {actual!r})')
    return stats

def _load(path):
    text = Path(path).read_text(encoding='utf-8')  # universal newlines
    suffix = Path(path).suffix
    if suffix == '.json':
        return json.loads(text)
    if suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == '.csv':
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return rows
        header = rows[0]
        for row in rows[1:]:
            if len(row) != len(header):
                raise AssertionError(f'{path}: malformed CSV row')
            for i, cell in enumerate(row):
                if not IDENTIFIER.search(header[i]) and cell.lower() in ('nan', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity'):
                    raise AssertionError(f'{path}: non-finite CSV numerical value')
                if not IDENTIFIER.search(header[i]) and DECIMAL.fullmatch(cell):
                    row[i] = float(cell)
        return rows
    raise ValueError(f'Unsupported comparison format: {suffix}')

def compare_files(expected, actual):
    stats = compare_values(_load(expected), _load(actual))
    return {'match': True, 'byte_identical': Path(expected).read_bytes() == Path(actual).read_bytes(),
            'relative_tolerance': RTOL, 'absolute_tolerance': ATOL, **stats}
