"""Record a new optional replay environment without launching a GPU workload."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/replay/code'))
from core import native_snapshot
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('Choose a new runtime record; do not overwrite an execution identity.')
args.output.write_text(json.dumps(native_snapshot(),indent=2)+'\n',encoding='utf-8')
print('Recorded runtime. No GPU job was launched.')
