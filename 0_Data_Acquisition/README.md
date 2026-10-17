# 0. From-scratch data acquisition

Normal team workflow: **start at step 1**. It uses local inputs or downloads missing agreed splits from
[the shared Drive folder](https://drive.google.com/drive/folders/1rkP9G4Bxp1xjhFW1COErvms-CMeNtJUd).

Use `download_split_process.ipynb` when reconstructing the data independently of Drive:

```bash
python run_all.py --from-stage 0 --to-stage 0
```

The notebook downloads the MIND-large train and labelled dev releases linked by the
[MIND website](https://msnews.github.io/), extracts raw files into `train/` and `test/`, then writes:

| Local folder | Split |
|---|---|
| `train/` | Earliest 85% of train-release impressions |
| `validation/` | Remaining 15% of train-release impressions |
| `test/` | Official labelled dev release, unchanged |

Each folder contains `behaviors_<split>.parquet` and `news_<split>.parquet`.
Entity vectors are kept in `train/` and `test/`; validation shares training vectors.
Original ZIPs are cached in `0_Data_Acquisition/downloads/`. All code/settings are inline.

Complete local splits are preserved by default. `--rebuild-splits` explicitly rebuilds them.
A stable timestamp sort makes rebuilds deterministic; the saved team assignment may differ within the
boundary second, so regenerate downstream datasets after rebuilding. The build marker prevents step 1
from mixing an interrupted rebuild with Drive files.

On 17 October 2026, the website's current download links returned HTTP 401 here and the old Azure links
returned HTTP 409. If access remains unavailable, update the inline URLs to working MIND website links,
or place downloaded `MINDlarge_train.zip` / `MINDlarge_dev.zip` in the downloads folder. Existing extracted
raw releases also work. This notebook cannot bypass source permissions.

Step 0 has one notebook: `download_split_process.ipynb`, containing download, extraction, and splitting.
