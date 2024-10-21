# warp

## authors

- **Connor Hainje**, NYU
- **David W Hogg**, NYU

## installation

```bash
pip install git+https://github.com/cmhainje/warp
```

## usage

```python
import warp

w = warp.Warp(scales=(W, H))  # W, H are width and height of your image
w.fit(control, target)  # control, target are (N_c, 2) and (N_t, 2) arrays containing your control and target points

w.transform(control)  # apply the warp to a set of points after fitting
```

see notebooks/demo.ipynb for more
