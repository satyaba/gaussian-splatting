import json

def load(p):
    with open(p) as f:
        return json.load(f)

van = load(r"D:\Backup\Downloads\grad_stats_vanilla.json")
sgd = load(r"D:\Backup\Downloads\grad_stats_run38.json")

def col(it, j):
    return [r[j] for r in it]

for name, d in (("VANILLA", van), ("SGDENSE", sgd)):
    it = d["iters"]
    n = col(it, 1)
    mean = col(it, 2)
    mx = col(it, 3)
    # detect abrupt n drops (opacity reset / prune)
    drops = [(it[i][0], n[i-1], n[i]) for i in range(1, len(it)) if n[i] < n[i-1] * 0.97]
    print("=" * 60)
    print(name, "iters rows:", len(it), "iter range", it[0][0], it[-1][0])
    print("  n: start %d  end %d  max %d" % (n[0], n[-1], max(n)))
    print("  mean grad: start %.3e  end %.3e  max %.3e" % (mean[0], mean[-1], max(mean)))
    print("  n-drop events (prune/reset):", [(i, a, b) for i, a, b in drops][:20])

# Do the two runs share early iterations? Compare n at same iter.
print("=" * 60)
print("Early-iteration agreement (iter, van_n, sgd_n, van_mean, sgd_mean, van_max, sgd_max):")
vd = {r[0]: r for r in van["iters"]}
sd = {r[0]: r for r in sgd["iters"]}
for i in [1, 2, 5, 10, 100, 500, 1000, 1500, 2000, 2100, 2500, 3000]:
    v = vd.get(i)
    s = sd.get(i)
    if v and s:
        print(f"{i:>5}  n={v[1]:>7}/{s[1]:>7}  mean={v[2]:.3e}/{s[2]:.3e}  max={v[3]:.3e}/{s[3]:.3e}")
