import json

def load(p):
    with open(p) as f:
        return json.load(f)

van = load(r"D:\Backup\Downloads\grad_stats_vanilla.json")
sgd = load(r"D:\Backup\Downloads\grad_stats_run38.json")

def summarize(name, d):
    ev = d["events"]
    it = d["iters"]
    print("=" * 70)
    print(name)
    print("  threshold       :", d["threshold"])
    print("  #event samples  :", len(ev), "  (logged every %d iters, from %d to %d)"
          % (ev[1]["iter"] - ev[0]["iter"], ev[0]["iter"], ev[-1]["iter"]))
    print("  #iters entries  :", len(it), " range", it[0][0], "->", it[-1][0])
    print("  first event:", {k: ev[0][k] for k in ("iter", "n", "mean", "median", "p90", "p99", "max", "frac_over")})
    print("  last  event:", {k: ev[-1][k] for k in ("iter", "n", "mean", "median", "p90", "p99", "max", "frac_over")})
    # peak mean
    pk = max(ev, key=lambda e: e["mean"])
    print("  peak mean grad : %.3e at iter %d (n=%d)" % (pk["mean"], pk["iter"], pk["n"]))
    # fraction over threshold stats
    fo = [e["frac_over"] for e in ev]
    print("  frac_over: min %.3f  max %.3f  mean %.3f" % (min(fo), max(fo), sum(fo) / len(fo)))
    print("  n(gaussians): first %d  last %d" % (ev[0]["n"], ev[-1]["n"]))
    # iters column interpretation
    print("  iters[0]:", it[0])
    print("  iters[-1]:", it[-1])

summarize("VANILLA 3DGS", van)
summarize("SGDense (run38)", sgd)

# ---- comparison over common iteration axis ----
print("=" * 70)
print("COMPARISON (aligned by iteration, events field)")
vd = {e["iter"]: e for e in van["events"]}
sd = {e["iter"]: e for e in sgd["events"]}
common = sorted(set(vd) & set(sd))
print("common event iters:", len(common), common[:5], "...", common[-3:] if common else "")
if common:
    print(f"{'iter':>6} {'van_mean':>11} {'sgd_mean':>11} {'ratio':>7} {'van_n':>8} {'sgd_n':>8} {'van_frac':>8} {'sgd_frac':>8}")
    for i in common:
        v, s = vd[i], sd[i]
        print(f"{i:>6} {v['mean']:>11.3e} {s['mean']:>11.3e} {s['mean']/v['mean']:>7.2f} "
              f"{v['n']:>8} {s['n']:>8} {v['frac_over']:>8.3f} {s['frac_over']:>8.3f}")
