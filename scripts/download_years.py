import os
import sys
import time
import pathlib
import shutil

# Configure SunPy directories inside workspace
cfg_dir = pathlib.Path("data/.sunpy").resolve()
cfg_dir.mkdir(parents=True, exist_ok=True)
mgr_dir = pathlib.Path("data/sunpy_manager").resolve()
mgr_dir.mkdir(parents=True, exist_ok=True)
dl_dir = pathlib.Path("data/downloads").resolve()
dl_dir.mkdir(parents=True, exist_ok=True)

sunpyrc = cfg_dir / "sunpyrc"
with open(sunpyrc, "w") as f:
    f.write(f"""[downloads]
download_dir = {dl_dir.as_posix()}
remote_data_manager_dir = {mgr_dir.as_posix()}
sample_dir = {dl_dir.as_posix()}
""")

os.environ["SUNPY_CONFIGDIR"] = str(cfg_dir)
os.environ["SUNPY_DOWNLOADDIR"] = str(dl_dir)

sys.path.insert(0, "third_party/ASR_cat")
from asr.utils import goes_downloader_old

for year in [2022, 2023]:
    out_file = f"data/raw/goes_{year}.csv"
    if os.path.exists(out_file):
        print(f"{out_file} already exists, skipping.", flush=True)
        continue
    print(f"[{time.strftime('%H:%M:%S')}] Starting download for year {year}...", flush=True)
    t0 = time.time()
    df = goes_downloader_old(year)
    df.to_csv(out_file, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Finished {year} in {time.time()-t0:.1f}s, saved {len(df):,} rows to {out_file}", flush=True)

if os.path.exists("data/downloads"):
    shutil.rmtree("data/downloads", ignore_errors=True)
print("All requested downloads complete!", flush=True)

