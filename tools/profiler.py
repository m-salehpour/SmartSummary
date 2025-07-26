import psutil, time, functools, os
from collections import namedtuple

# A small helper to snapshot stats
Snapshot = namedtuple("Snapshot", ["ts", "cpu_user", "cpu_sys", "rss", "vms", "read_bytes", "write_bytes"])

def take_snapshot():
    p = psutil.Process(os.getpid())
    io = psutil.disk_io_counters()
    cpu_times = p.cpu_times()
    mem = p.memory_info()
    return Snapshot(
        ts=time.time(),
        cpu_user=cpu_times.user,
        cpu_sys=cpu_times.system,
        rss=mem.rss,
        vms=mem.vms,
        read_bytes=io.read_bytes,
        write_bytes=io.write_bytes,
    )

def profile_resources(func):
    """Decorator to measure resources used by a single call to func(...)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        before = take_snapshot()
        result = func(*args, **kwargs)
        after = take_snapshot()
        # compute deltas
        wall  = after.ts - before.ts
        cpu   = (after.cpu_user + after.cpu_sys) - (before.cpu_user + before.cpu_sys)
        rss   = (after.rss - before.rss) / (1024**2)        # MB
        vms   = (after.vms - before.vms) / (1024**2)        # MB
        read  = (after.read_bytes - before.read_bytes) / (1024**2)  # MB
        write = (after.write_bytes - before.write_bytes) / (1024**2) # MB
        print(f"\n🔍 Resource profile for `{func.__name__}`:")
        print(f"  Wall-time:   {wall:.1f}s")
        print(f"  CPU time:    {cpu:.1f}s")
        print(f"  RSS Δ:       {rss:.1f} MB")
        print(f"  VMS Δ:       {vms:.1f} MB")
        print(f"  Disk Read:   {read:.1f} MB")
        print(f"  Disk Write:  {write:.1f} MB\n")
        return result
    return wrapper
