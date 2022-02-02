# Software for Fitting the Upscaled and Extended Model of Capillary Rise to Experimental Data

This Python based software is used to produce the fitting reported by
[Lunowa, Mascini, et al. (2022)](#references). It requires the data from
[Heshmati and Piri (2014)](#references), but can be simply extended to include
other data sets, where the rise height and the contact angle are reported over
time.

Note that the **data reported by [Heshmati and Piri (2014)](#references) is not
included** due to legal restrictions (copyright). Instead, for demonstration
purposes, the included data (`data/<fluid>_R<radius>mm<#>.csv`) consists of samples
produced from the fitted models.
However all the fits (`data/<fluid>_R<radius>mm.fit`), the parameter data
(`data/<fluid>_R<radius>mm_<modeltype>_<fittype><slip>.param`) and the summaries
(`<fluid>_<modeltype>.csv`), which are reported in
[Lunowa, Mascini, et al. (2022)](#references), are included.


## Installation

This software requires no installation, but has several dependencies. First,
make sure you have Python (version 3.6 or higher) available from your command
line. You can check this by running:

| Unix/macOS:             | Windows:          |
|-------------------------|-------------------|
| `python3 --version`     | `py --version`    |

You should get some output like `Python 3.8.10`. If you do not have Python,
please install the latest 3.x version from [python.org](https://www.python.org).
Additionally, you need to make sure you have pip available. You can check this
by running:

| Unix/macOS:                    | Windows:                 |
|--------------------------------|--------------------------|
| `python3 -m pip --version`     | `py -m pip --version`    |

You should get some output like `pip 21.1.3 [...]`. If you do not have pip, please
install the latest version as described in the
[online tutorial](https://packaging.python.org/tutorials/installing-packages/).
Finally, pip can install the required python3 packages
[scipy](https://www.scipy.org/), [numpy](https://numpy.org/),
[lmfit](https://lmfit.github.io/lmfit-py/), [pandas](https://pandas.pydata.org/),
[matplotlib](https://matplotlib.org/) and [notebook](https://jupyter.org/index.html).
For this, you can run (in this directory):

| Unix/macOS:                                      | Windows:                                   |
|--------------------------------------------------|--------------------------------------------|
| `python3 -m pip install -r requirements.txt`     | `py -m pip install -r requirements.txt`    |


## Usage

To use the included jupyter notebook (`DataVisualization.ipynb`), you need to
execute the following command (in this directory):
```
jupyter notebook DataVisualization.ipynb
```
Then a browser tab will open, where the notebook is started. In it, the steps
to perform the fitting and to visualize the results are explained.

To only use the fitting procedure, you can also execute the following command
(in this directory):

| Unix/macOS:                   | Windows:                |
|-------------------------------|-------------------------|
| `python3 DataAnalysis.py`     | `py DataAnalysis.py`    |

This runs the fitting (method `fit_and_save`) for all 3 fluids (glycerol,
Soltrol 170, purified water). **Be aware that this might take several hours.**

The details of the implementation for the data analysis can be found in the file
`DataAnalysis.py`, the visualization in `DataVisualization.py`.
Note the following parameters, which could slightly affect the fitting procedure:

* When the experimental data is read in the function `read_data_set`, the
  arguments `h_rel` (default 0.9) and `t_rel` (default 0.2) define the minimal
  relative rise height and minimal relative time after which the data values are
  used to extract the static contact angle.
* The function `_extended_model_integrator` for the solution of the extended
  model has an dict argument `kws` that may include the initial velocity `"v0"`
  (default described by [Lunowa, Mascini, et al. (2022)](#references)) and the
  maximal time step for the numerical integration of the ODE `"max_step"`
  (default 0.01).
* The function `fit_and_save(<fluid>, kws)` has a dict argument `kws` that may
  contain the key `"t_max"` setting the maximal (nondimensional) time for the
  extended model fit (function `fit_eta_slip`) after which the experimental data
  is ignored to reduce the time required by the repeated numerical solution.
  The default value is infinite (no data ignored) except for soltrol (400, when
  the stationary rise height is already reached).


## Adding and fitting other experimental data sets

To add experimental data sets, the data of each experiment must be included in a
file `data/<fluid>_R<radius>mm<#>.csv` where `<fluid>` is the name of the fluid
(here: glycerol, soltrol, water), `<radius>` is the tube radius and `<#>` is the
number of the experiment. This file should contain comma separated values (CSV)
labeled `rise,time,CA` (the rise height in cm, the time in seconds, and the
contact angle in degrees measured in the rising fluid). It may start by a
comment line beginning with `#`. For an example, see the file
`data/glycerol_R0.5mm1.csv`.

To fit the model for an fluid with other properties than those reported in
[Lunowa, Mascini, et al. (2022)](#references), you need to change the file
`DataAnalysis.py` by the following steps:

1. In the dict `_fit_extended` add for the fluid whether the extended model
   shall be fitted (or only the upscaled model).
2. In the function `get_parameters`, the following parameters of the fluid must
   be set.
   - `sigma`: the surface tension in kg/s^2
   - `mu`: the viscosity of the rising fluid in kg/m/s
   - `rho`: the density of the rising fluid in kg/m^3
   - `max_eta`: the maximal dynamic parameter in the fitting (nondimensional)
3. In the function `fit_and_save`, the radii of the used tubes in mm must be set.
4. In the end after `if __name__ == "__main__":` add the fitting for the fluid
   as for the other fluids `fit_and_save(<fluid>, kws)`. The dict `kws` may
   contain the key `"t_max"` setting the maximal (nondimensional) time for the
   extended model after which the experimental data is ignored to reduce the
   time required by the repeated numerical solution (e.g. 400 for soltrol after
   which the experiments reached the stationary rise height).

To visualize the results, add and run a code block in the notebook
`DataVisualization.ipynb`, just as for the other fluids.


## References

The software is based on the following publications:

* M. Heshmati, M. Piri, *Experimental investigation of dynamic contact angle and
  capillary rise in tubes with circular and noncircular cross sections*.
  Langmuir 30 (2014) 14151-14162.
  [DOI: 10.1021/la501724y](https://doi.org/10.1021/la501724y).
* S.B. Lunowa, C. Bringedal, I.S. Pop, *On an averaged model for immiscible
  two-phase flow with surface tension and dynamic contact angle in a thin strip*.
  Stud. Appl. Math. 147 (2021) 84-126.
  [DOI: 10.1111/sapm.12376](https://doi.org/10.1111/sapm.12376)
* S.B. Lunowa, A. Mascini, C. Bringedal, T. Bultreys, V. Cnudde, I.S. Pop,
  *Dynamic effects during the capillary rise of fluids in cylindrical tubes*.
  Langmuir (2022). [DOI: 10.1021/acs.langmuir.1c02680](https://doi.org/10.1021/acs.langmuir.1c02680).


## Copyright, License & Contact

Copyright (c) 2021 Stephan B. Lunowa

This software is licensed under the [MIT License](./LICENSE).

In case of any questions regarding this software you can get into contact via
[stephan.lunowa@uhasselt.be](mailto:stephan.lunowa@uhasselt.be).
