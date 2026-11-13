# sfincs-runs

Working branch for the SFINCS sea-flood runs (today, 2036, 2046 × SSP2-4.5/5-8.5 × sinking continues/stops × with/without dikes).
`runs/<name>/` are the model inputs written by `sf_setup.py` (sinking rate gap-filled with the land-cover model, `scripts/vlm_fill.py`);
GitHub Actions runs each one with the official `deltares/sfincs-cpu:sfincs-v2.5.0-Hautacam` image and commits `results/<name>/hmax.npz`.
The maps, tables and web layers are built from these results on the main branches. This branch can be deleted afterwards.
