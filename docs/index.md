<p align="center">
  <img src="https://raw.githubusercontent.com/amanchokshi/skyweaver/main/docs/imgs/skyweaver.gif" width=100%>
</p>

**A framework for simulating how satellite orbits weave across telescope skies.**

---

## What is skyweaver?

`skyweaver` is a Python framework for modelling how satellite orbits interact with telescope observations:

- compute **ground tracks** across Earth  
- generate **sky tracks** for specific observatories  
- sample the sky in **alt–az coordinates**  
- accumulate **HEALPix sky coverage maps**

---

## Core workflow

```python
orbit → timegrid → ground_track → sky_track → healpix
```
