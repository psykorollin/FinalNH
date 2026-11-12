"""เติมช่องว่างของอัตราทรุด (InSAR) ด้วย IDW ให้เต็มพื้นที่ศึกษา (สามจังหวัด)

อินพุต (จาก site เดิม): grid.bin (int16 0.1 มม./ปี บน lon/lat), unc.bin (+10000 = จุดที่แสดงจาง), districts.geojson, meta.json
เอาต์พุต:
- vlm_idw.png  ภาพ 5 ระดับ EPSG:3857 เฉพาะช่องที่เติม (ใช้ซ้อนใต้ vlm.png)
- idw.bin      int16 สองค่าต่อช่อง: อัตราที่เติม (0.1 มม./ปี; -32768 = ไม่ได้เติม) และระยะถึงจุดวัดใกล้สุด (×100 ม.)
- meta.json    เพิ่ม meta.idw = {image, power, k, n_src, n_fill, ...}

    python scripts/idw_fill.py [data/web/site]   # รันหลัง scripts/webmap.py

IDW: power 2, จุดวัดใกล้สุด 12 จุด (เฉพาะจุดที่ไม่แสดงจาง) ระยะคิดบนระนาบ (ลองจิจูด × cos ละติจูด)
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import unary_union

SITE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'data' / 'web' / 'site'


def to3857(lon, lat):
    import math
    R = 6378137.0
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


POWER, K = 2, 12
EDGES = [0, -5, -10, -20]
ALPHA = 200

meta = json.loads((SITE / 'meta.json').read_text())
gr = meta['grid']
h, w = gr['h'], gr['w']
gt = Affine(gr['dx'], 0, gr['x0'], 0, gr['dy'], gr['y0'])
q = np.fromfile(SITE / 'grid.bin', '<i2').reshape(h, w)
u = np.fromfile(SITE / 'unc.bin', '<i2').reshape(h, w)
dist = json.loads((SITE / 'districts.geojson').read_text(encoding='utf-8'))
area = unary_union([shape(f['geometry']) for f in dist['features']])
inside = rasterize([area], out_shape=(h, w), transform=gt, fill=0, all_touched=True).astype(bool)

has = q != -32768
faded = has & (u != -32768) & (u >= 10000)
src = has & ~faded
gap = inside & ~has
print(f'ในพื้นที่ศึกษา {inside.sum():,} ช่อง | มีค่า {(inside & has).sum():,} | ช่องว่าง {gap.sum():,} '
      f'({100 * gap.sum() / inside.sum():.1f}%) | จุดต้นทาง {src.sum():,}')

coslat = np.cos(np.deg2rad(gr['y0'] + gr['dy'] * h / 2))
rr, cc = np.nonzero(src)
xy = lambda r, c: np.c_[c * gr['dx'] * coslat, r * -gr['dy']]
tree = cKDTree(xy(rr, cc))
vals = q[rr, cc].astype('float64') / 10
gr_, gc_ = np.nonzero(gap)
d, i = tree.query(xy(gr_, gc_), k=K, workers=-1)
d_m = d * 111320.0
wgt = 1.0 / np.maximum(d_m, 1.0) ** POWER
est = (wgt * vals[i]).sum(1) / wgt.sum(1)
near = d_m[:, 0]
print(f'ระยะถึงจุดวัดใกล้สุด: มัธยฐาน {np.median(near):.0f} ม. | p95 {np.percentile(near, 95):.0f} ม. | สูงสุด {near.max():.0f} ม.')

# leave-one-out check on a sample of measured points (neighbours exclude the point itself)
rng = np.random.default_rng(0)
smp = rng.choice(len(rr), 20000, replace=False)
d2, i2 = tree.query(xy(rr[smp], cc[smp]), k=K + 1, workers=-1)
d2, i2 = d2[:, 1:] * 111320.0, i2[:, 1:]
w2 = 1.0 / np.maximum(d2, 1.0) ** POWER
e2 = (w2 * vals[i2]).sum(1) / w2.sum(1)
err = e2 - vals[smp]
print(f'LOO (จุดวัดจริง 20,000 จุด): MAE {np.abs(err).mean():.2f} RMSE {np.sqrt((err ** 2).mean()):.2f} มม./ปี')

fill = np.full((h, w), np.nan, 'float32')
fill[gr_, gc_] = est
out = np.full((h, w, 2), -32768, '<i2')
out[gr_, gc_, 0] = np.round(est * 10)
out[gr_, gc_, 1] = np.clip(np.round(near / 100), 0, 32000)
out.tofile(SITE / 'idw.bin')

# image on the exact EPSG:3857 grid of vlm.png: full surface (measured + IDW), painted only where vlm.png is empty
from rasterio.transform import from_bounds
vimg = Image.open(SITE / 'vlm.png').convert('RGBA')
vw, vh = vimg.size
(lx0, ly1), _, (lx1, ly0), _ = meta['image']
X0, Y1 = to3857(lx0, ly1); X1, Y0 = to3857(lx1, ly0)
dt = from_bounds(X0, Y0, X1, Y1, vw, vh)
surf = np.where(has, q / 10.0, np.nan).astype('float32')
surf[gr_, gc_] = est
surf[~inside] = np.nan
m = np.full((vh, vw), np.nan, 'float32')
reproject(surf, m, src_transform=gt, src_crs='EPSG:4326', src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
          dst_nodata=np.nan, resampling=Resampling.nearest)
cls = np.full(m.shape, -1, 'int8')
ok = np.isfinite(m)
cls[ok] = 0
for k_, e in enumerate(EDGES, start=1):
    cls[ok & (m < e)] = k_
rgba = np.zeros((*m.shape, 4), 'uint8')
for k_, col in enumerate(meta['colors']):
    rgba[cls == k_] = [int(col[j:j + 2], 16) for j in (1, 3, 5)] + [ALPHA]
rgba[np.asarray(vimg)[..., 3] > 0] = 0
Image.fromarray(rgba).save(SITE / 'vlm_idw.png', optimize=True)
x0, y0, x1, y1 = lx0, ly0, lx1, ly1
meta['idw'] = {'image': [[x0, y1], [x1, y1], [x1, y0], [x0, y0]], 'power': POWER, 'k': K,
               'n_src': int(src.sum()), 'n_fill': int(gap.sum()),
               'pct_fill': round(100 * float(gap.sum()) / float(inside.sum()), 1),
               'loo_mae': round(float(np.abs(err).mean()), 2), 'loo_rmse': round(float(np.sqrt((err ** 2).mean())), 2),
               'near_med_m': int(np.median(near)), 'near_p95_m': int(np.percentile(near, 95))}
(SITE / 'meta.json').write_text(json.dumps(meta))
print(json.dumps(meta['idw']))
