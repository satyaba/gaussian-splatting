import json, math

van = json.load(open(r"D:\Backup\Downloads\grad_stats_vanilla.json"))
sgd = json.load(open(r"D:\Backup\Downloads\grad_stats_sgdense.json"))

vd = {e["iter"]: e for e in van["events"]}
sd = {e["iter"]: e for e in sgd["events"]}
iters = sorted(set(vd) & set(sd))
x0, x1 = iters[0], iters[-1]

W, H = 1000, 720
padL, padR, padT = 80, 30, 50
panelH = 260
gap = 60

def sx(i):
    return padL + (i - x0) / (x1 - x0) * (W - padL - padR)

def poly(vals, ymin, ymax, y0):
    def sy(v):
        v = max(v, ymin)
        return y0 + panelH - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * panelH
    pts = " ".join(f"{sx(i):.1f},{sy(vals[i]):.1f}" for i in iters)
    return pts, sy

parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI,Arial" font-size="13">')
parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def panel(title, key, ymin, ymax, y0, fmt):
    p = []
    p.append(f'<text x="{padL}" y="{y0-12}" font-size="16" font-weight="600">{title}</text>')
    # gridlines
    e = math.log10(ymin)
    while e <= math.log10(ymax) + 1e-9:
        v = 10 ** e
        yy = y0 + panelH - (e - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * panelH
        p.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" stroke="#eee"/>')
        p.append(f'<text x="{padL-8}" y="{yy+4:.1f}" text-anchor="end" fill="#555">{fmt(v)}</text>')
        e += 1
    vv = {i: vd[i][key] for i in iters}
    ss = {i: sd[i][key] for i in iters}
    pv, syv = poly(vv, ymin, ymax, y0)
    ps, sys = poly(ss, ymin, ymax, y0)
    p.append(f'<polyline points="{pv}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    p.append(f'<polyline points="{ps}" fill="none" stroke="#d62728" stroke-width="2"/>')
    return "\n".join(p)

# panel 1: mean gradient (log)
parts.append(panel("Mean gradient magnitude vs iteration", "mean", 5e-6, 2e-3, padT, lambda v: f"{v:.0e}"))
# panel 2: fraction over densify threshold 2e-4
parts.append(panel("Fraction of Gaussians with |grad| > 2e-4 (densify candidate rate)", "frac_over", 1e-3, 1.0, padT + panelH + gap, lambda v: f"{v:g}"))

# x-axis labels on bottom panel
yax = padT + 2 * panelH + gap
for i in range(x0, x1 + 1, 1000):
    xx = sx(i)
    parts.append(f'<line x1="{xx:.1f}" y1="{yax}" x2="{xx:.1f}" y2="{yax+5}" stroke="#555"/>')
    parts.append(f'<text x="{xx:.1f}" y="{yax+20}" text-anchor="middle" fill="#555">{i}</text>')
parts.append(f'<text x="{(padL+W-padR)/2}" y="{yax+42}" text-anchor="middle">iteration</text>')

# legend
lx = W - 240
parts.append(f'<rect x="{lx}" y="12" width="14" height="14" fill="#1f77b4"/><text x="{lx+20}" y="24">Vanilla 3DGS</text>')
parts.append(f'<rect x="{lx+120}" y="12" width="14" height="14" fill="#d62728"/><text x="{lx+140}" y="24">SGDense</text>')
parts.append('</svg>')

open(r"D:\Projects\Personal\2026\SGDense-3DGS\grad_compare.svg", "w").write("\n".join(parts))
print("wrote grad_compare.svg")
