# Notebook builders

The notebooks in `1_Data_Preparation/` and `2_EDA/` are generated from these scripts (nbformat) so that text, code and
figure conventions stay in one place. Editing the `.ipynb` files directly also works; regenerate only if you change a builder.

```
python3 notebook_builders/build_prep.py        # -> 1_Data_Preparation/clean_and_validate_data.ipynb
python3 notebook_builders/build_impression.py  # -> 2_EDA/impression_analysis.ipynb
python3 notebook_builders/build_user.py        # -> 2_EDA/user_analysis.ipynb
python3 notebook_builders/build_article.py     # -> 2_EDA/article_temporal_analysis.ipynb
cd 1_Data_Preparation && jupyter nbconvert --to notebook --execute --inplace clean_and_validate_data.ipynb
cd ../2_EDA && jupyter nbconvert --to notebook --execute --inplace impression_analysis.ipynb   # same for the other two
```

`verify_split.py` is the stand-alone script used on 14 Oct 2026 to double-check the split before the notebook existed;
its checks are now part of the data-preparation notebook.
