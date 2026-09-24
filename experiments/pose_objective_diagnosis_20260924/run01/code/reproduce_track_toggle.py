"""Run the frozen track_toggle implementation under a fresh output root."""
from pathlib import Path
import argparse,sys
import track_toggle
ap=argparse.ArgumentParser();ap.add_argument('--new-root',type=Path,required=True);a=ap.parse_args()
new=a.new_root.resolve()
if new.exists():raise RuntimeError('A genuinely new root is required; refusing overwrite')
track_toggle.NEW=new
sys.argv=[track_toggle.__file__,'--output',str(new/'track_toggle')]
track_toggle.main()
