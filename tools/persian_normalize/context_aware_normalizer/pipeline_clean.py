#!/usr/bin/env python3
# pipeline_clean.py
#
# 1. read WhisperX JSON
# 2. summarise entire text  (cached)
# 3. clean every segment with SegmentCleaner
# 4. write result JSON

import json, sys, time, asyncio
from pathlib import Path
from tools.persian_normalize.context_aware_normalizer.summarise import get_summary
from tools.persian_normalize.context_aware_normalizer.normalise_segments import SegmentCleaner


def main(inp: Path, outp: Path):
    print(f"🗂  loading {inp}")
    raw = json.loads(inp.read_text(encoding="utf-8"))
    segs = raw["segments"] if isinstance(raw, dict) else raw
    for i, s in enumerate(segs, 1):
        s["_idx"] = i

    # ------------------------------------------------------------------
    meta = get_summary(inp)
    print("📄 summary cached / generated:")
    print(meta["summary"])
    print("—" * 60)

    # ------------------------------------------------------------------
    cleaner = SegmentCleaner(summary=meta["summary"])
    asyncio.run(cleaner.clean_all(segs))

    # ------------------------------------------------------------------
    outp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 wrote {outp}")

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage:  python pipeline_clean.py in.json out.json"); sys.exit(1)
    t0 = time.time()
    main(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"⏱  total {time.time() - t0:.1f}s")
