# pyanodi

pyanodi is a Python package to perform the analysis of distance designed by [Tan et al. (2014)](https://doi.org/10.1007/s11004-013-9482-1) to compare geostatistical simulation algorithms.

## Disclaimer

pyanodi was developed to work with continuous variables in 2D, so all the options mentioned in the paper haven't been implemented. It can only handle 2D images for now, and contains the cluster-based histograms of patterns but not the multiple-point histograms.

## Installation

You can directly install pyanodi from GitHub using pip:

    pip install git+https://github.com/grongier/pyanodi.git

## Use

```
from pyanodi import ANODI
from sklearn.manifold import MDS

training_image = ... # array of shape (n_cells_ti_y, n_cells_ti_x)
realizations = ...   # array of shape (n_methods, n_realizations_per_method, n_cells_y, n_cells_x)

# Set the parameters
anodi = ANODI(pyramid=(1, 2, 3),
              halfwindow=15,
              max_halfwindow=None,
              n_clusters=48,
              step=2,
              use_mds=False,
              method='mean',
              verbose=True,
              random_state=100,
              n_jobs=4)

# Perform the analysis
anodi.fit_transform(training_image, realizations)

# Get the MDS representation of the distances between the images
pyramid_level = 0
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=100)
mds_points = mds.fit_transform(anodi.distances_[..., pyramid_level])

# Get the rankings of the methods
anodi.score()
```

## Citation

If you use pyanodi in your research, please cite the original article:

> Tan, X., Tahmasebi, P. & Caers, J. (2014). Comparing Training-Image Based Algorithms Using an Analysis of Distance. *Mathematical Geosciences*, 46(2), 149-169. doi:10.1007/s11004-013-9482-1

Here is the corresponding BibTex entry if you use LaTex:

    @Article{Tan2014,
        author="Tan, Xiaojin
        and Tahmasebi, Pejman
        and Caers, Jef",
        title="Comparing Training-Image Based Algorithms Using an Analysis of Distance",
        journal="Mathematical Geosciences",
        year="2014",
        volume="46",
        number="2",
        pages="149--169",
        issn="1874-8953",
        doi="10.1007/s11004-013-9482-1",
    }
