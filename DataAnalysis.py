import os
import re

import numpy
import scipy.optimize
import scipy.integrate
import pandas
import lmfit

# result types of fitting the height, the contact angle, or both
residual_types = ["h", "ca", "b"]

# whether the extended model is fitted for the fluids (or only the upscaled one)
_fit_extended = {
  "glycerol": False, # do not fit the extended model for glycerol
  "soltrol": True,
  "water": True
}

################################################################################
#   PARAMETER GENERATOR AND MODELS
################################################################################

def get_parameters(fluid, radius, theta_s):
    """Returns a lmfit.Parameters object containing all necessary parameters for
       the given `fluid`, `radius` [m] and static contact angle `theta_s` [rad]
       (measured in the receding fluid 2).
    """
    if fluid == "glycerol":
        sigma = 6.347e-2 # 63.47 dyne/cm = 6.347e-2 kg/s**2
        mu = 1.0111      # 10.111 Poise = 1.0111 kg/m/s
        rho = 1260       # 1.26 g/cm^3 = 1260 kg/m**3
        max_eta = 15
    elif fluid == "soltrol":
        sigma = 2.483e-2 # 24.83 dyne/cm = 2.483e-2 kg/s**2
        mu = 2.6e-3      # 0.026 Poise = 2.6e-3 kg/m/s
        rho = 774        # 0.774 g/cm^3 = 774 kg/m**3
        max_eta = 10
    elif fluid == "water":
        sigma = 7.28e-2  # 72.8 dyne/cm = 7.28e-2 kg/s**2
        mu = 1.1e-3      # 0.011 Poise = 1.1e-3 kg/m/s
        rho = 997        # 0.997 g/cm^3 = 997 kg/m**3
        max_eta = 10
    else:
        raise NotImplementedError

    parameters = lmfit.Parameters()
    # parameters of the dimensional problem
    parameters.add("g", value=9.81, vary=False)      # gravity         [m / s**2]
    parameters.add("sigma", value=sigma, vary=False) # surface tension [kg / s**2]
    parameters.add("mu", value=mu, vary=False)       # viscosity       [kg / m / s]
    parameters.add("rho", value=rho, vary=False)     # density         [kg / m**3]
    parameters.add("R_0", value=radius, vary=False)  # tube radius     [m]
    parameters.add("theta_s", value=theta_s, vary=False) # static CA, nondimensional
    # scaling parameters
    parameters.add("L", expr="-2*sigma*cos(theta_s)/(rho*g*R_0)") # Length scale: Jurin's height [m]
    parameters.add("U", expr="R_0**2*rho*g/mu")      # Velocity scale [m/s]
    parameters.add("T", expr="L/U")                  # Time scale [s]
    parameters.add("P", expr="rho*g*L")              # Pressure scale [kg/m/s^2]
    # dinemsionless numbers
    parameters.add("epsilon", expr="R_0/L")          # scale separation, aspect ratio
    parameters.add("Re", expr="rho*U*L/mu")          # effective Reynolds number
    parameters.add("I", expr="epsilon**2 * Re")      # Inertia term, if relevant
    parameters.add("Ca", expr="mu*U/sigma/epsilon")  # effective capillary number (scaled by epsilon)
    # model parameters
    parameters.add("eta", value=0.0, min=0.0, max=max_eta) # dynamic CA-vel coeff., nondimensional
    parameters.add("slip", value=0.0, min=0.0, max=5.0, vary=False) # slip, nondimensional
    return parameters


def height_theta(parameters, time, extended=False):
    """Returns the height and the contact angle at the given times for the upscaled/extended model."""
    if extended:
      height, velocity = _extended_model_integrator(parameters, time)
      theta = _theta(parameters, velocity)
    else:
      height = _height_upscaled(parameters, time)
      theta = _theta(parameters, (1 - height) / ( 8/(4*parameters["slip"]+1)*height + 2*parameters["eta"] ))
    return height, theta


def _height_upscaled(parameters, time):
    """Returns the non-dimensional height at the given time for the upscaled model."""
    if type(time) is float:
        if time < 0.0:
            height = 0.0
        elif time * parameters["epsilon"]**2 > 1.0:
            height = 1.0
        else:
            slip = parameters["slip"]
            denom = 1.0 + (slip + 0.25) * parameters["eta"]
            fun = lambda x : 1 - (1 - x) * numpy.exp(x / denom + (4*slip + 1) / (8*denom) * time)
            try:
                sol = scipy.optimize.root_scalar(fun, bracket=[0, 1])
                height = sol.root if sol.converged else numpy.nan
            except:
                height = numpy.nan
    else:
        height = time.apply(lambda t : _height_upscaled(parameters, t))
    return height


def _theta(parameters, velocity):
    """Returns the contact angle for the given velocity."""
    return numpy.arccos(numpy.cos(parameters["theta_s"]) + parameters["eta"] * parameters["Ca"] * velocity)


def _extended_model_integrator(parameters, times, **kws):
    """Solver for the extended model by reformulation and using SciPy integrator"""
    # correct time order
    times, mask, order = _ordered_array(times)

    # parameters
    I = parameters["I"].value
    B = 4.0 / (4.0 * parameters["slip"] + 1.0)
    eta = parameters["eta"].value
    D = 1.0/numpy.cos(parameters["theta_s"])

    # initial values
    y0 = numpy.array([1e-12, 2e-6 * kws.pop("v0", (numpy.sqrt(eta**2 + I) - eta) / I)])

    # RHS of d^2/dt^2 w = 2/I * ( 1 - sqrt(w) - ( B + eta/sqrt(w) ) * d/dt w )
    def RHS(t, w_dwdt):
        w, dw_dt = w_dwdt
        return numpy.asarray([dw_dt, 2/I * (1 - numpy.sqrt(w) - B * dw_dt - min(1-D, max(1+D, eta * dw_dt/numpy.sqrt(w))))])
    def JAC(t, w_dwdt):
        w, dw_dt = w_dwdt
        val = eta * dw_dt/numpy.sqrt(w)
        return numpy.asmatrix([[0, 1], [( -1/numpy.sqrt(w) + (val > 1+D and val < 1-D) * eta*dw_dt/(w**1.5) ) / I,
                                        -2/I*( B + (val > 1+D and val < 1-D) * eta/numpy.sqrt(w) )]])
    # solve and transform
    if "max_step" not in kws: kws["max_step"] = 1e-2
    res = scipy.integrate.solve_ivp(RHS, [0, times[-1]], y0, method="BDF", t_eval=times, jac=JAC ,**kws)
    height = numpy.sqrt(res.y[0, mask][order])
    velocity = res.y[1, mask][order] / (2*height)
    return height, velocity


################################################################################
#   DATA FITTING
################################################################################

def residual(parameters, time, data_h=None, data_ca=None, extended=False):
    """Returns the scaled residual between upscaled/extended model and data for fitting
       the height and/or the contact angle
    """
    height_scaling = 20; theta_scaling = 20
    height, theta = height_theta(parameters, time, extended)
    if data_h is None:
        return theta_scaling * (data_ca - theta)
    elif data_ca is None:
        return height_scaling * (data_h - height)
    else:
        return (height_scaling * (data_h - height)).append(theta_scaling * (data_ca - theta))


def fit_eta_slip(parameters, data, extended=False, t_max=numpy.inf):
    """Returns the best-fits of the dynamic coefficient and the slip for the
       upscaled/extended model wrt. the data up to time `t_max` using the
       height, the contact angle, and both.
    """
    results = {s : [] for s in residual_types}
    data_t = data["t"]; data_h = data["h"]; data_ca = data["theta"]
    mask = data_t.to_numpy() < t_max
    nnz = numpy.count_nonzero(numpy.logical_not(mask))
    if nnz > 0: print("Maximal time T = {}, {} data points disregarded.".format(t_max, nnz))

    # only eta is parameter (slip = 0.0)
    parameters["eta"].set(vary=True, value=1.0) # initial eta for fitting must not be zero
    parameters["slip"].set(vary=False, value=0.0)
    results["h"].append(lmfit.minimize(residual, parameters, args=(data_t[mask], data_h[mask], None, extended),
                                       nan_policy='omit', method="nelder"))
    results["ca"].append(lmfit.minimize(residual, parameters, args=(data_t[mask], None, data_ca[mask], extended),
                                        nan_policy='omit', method="nelder"))
    results["b"].append(lmfit.minimize(residual, parameters, args=(data_t[mask], data_h[mask], data_ca[mask], extended),
                                       nan_policy='omit', method="nelder"))
    # eta and slip are parameter
    parameters['slip'].set(vary=True, value=0.1) # initial slip for fitting must not be zero
    results["h"].append(lmfit.minimize(residual, parameters, args=(data_t[mask], data_h[mask], None, extended),
                                       nan_policy='omit', method="nelder"))
    results["ca"].append(lmfit.minimize(residual, parameters, args=(data_t[mask], None, data_ca[mask], extended),
                                        nan_policy='omit', method="nelder"))
    results["b"].append(lmfit.minimize(residual, parameters, args=(data_t[mask], data_h[mask], data_ca[mask], extended),
                                       nan_policy='omit', method="nelder"))
    return results


def fit_and_save(fluid, kws={}):
    """Fits and saves the results for the given fluid in the files `data/<fluid>_R<radius>mm.fit`
       and `data/<fluid>_R<radius>mm_<modeltype>_<residual_types><#>.param`.
    """
    if fluid == "glycerol":
        radii = [0.25, 0.5, 1.0] # radius in mm
    elif fluid == "soltrol":
        radii = [0.375, 0.5, 0.65] # radius in mm
    elif fluid == "water":
        radii = [0.375, 0.5, 0.65] # radius in mm
    else:
        raise NotImplementedError
    data, parameters = read_data_set(fluid, radii)
    fits_upscaled = []; fits_extended = []
    for i, radius in enumerate(radii):
        fits_upscaled.append( fit_eta_slip(parameters[i], data[i], extended=False) )
        if _fit_extended[fluid]:
            fits_extended.append( fit_eta_slip(parameters[i], data[i], extended=True, **kws) )
        with open(os.path.join("data", fluid + "_R" + str(radius) + "mm.fit"), 'w') as f:
            _write_fits_parameters(f, fits_upscaled[i], fluid, radius, "ups")
            if _fit_extended[fluid]: _write_fits_parameters(f, fits_extended[i], fluid, radius, "ext")

    _write_fit_summary(fluid + "_ups.csv", radii, fits_upscaled)
    if _fit_extended[fluid]: _write_fit_summary(fluid + "_ext.csv", radii, fits_extended)


################################################################################
#   DATA IO
################################################################################

def read_data_set(fluid, radii, h_rel = 0.9, t_rel = 0.2):
    """Reads all the data of the given fluid with given list `radii` from the files
       `data/<fluid>_R<radius>mm<#>.csv` and includes it into a list of DataFrames.
       Additionally, nondimensional quantities are computed and the static contact
       angle is extracted and returned together with the fluid and radius specific
       parameters.
       The read files must be in CSV format and include the columns "time", "rise"
       and "CA".

       The returned DataFrames contain the following columns:
           * "time" [s]
           * "rise" height [cm]
           * "CA" contact angle (measured in the rising fluid) [deg]
           * "theta" contact angle (measured in the receding fluid 2) [rad]
           * "t" nondimensional time
           * "h" nondimensional rise height
    """
    data = []
    theta_s = []
    for i, radius in enumerate(radii):
        files = []
        for filename in os.listdir("data"):
            regex = fluid + "_R" + str(radius) + "mm\d.csv"
            if re.fullmatch(regex, filename) is not None:
                files.append(os.path.join("data", filename))

        data.append(_read_data(files))
        # extract data at late time and high rise
        theta_s.append( data[i]['CA'][( data[i]['rise'] > h_rel * numpy.max(data[i]['rise']) )
                                    & ( data[i]['time'] > t_rel * numpy.max(data[i]['time']) )] )

    theta_s = pandas.concat(theta_s, ignore_index=True)
    print("Static contact angle {:.2f} degrees extracted from {:d} values for {}."
          .format(numpy.average(theta_s), theta_s.size, fluid))

    parameters = [];
    for i, radius in enumerate(radii):
        parameters.append( get_parameters(fluid, radius*1e-3, numpy.pi - numpy.deg2rad(numpy.average(theta_s))) )
        data[i]["t"] = data[i]["time"] / parameters[i]['T']
        data[i]["h"] = data[i]["rise"] / parameters[i]['L'] / 100 # rise given in [cm]

    return data, parameters


def load_fits(fluid, radii):
    """Loads the saved parameters from fitting of the given fluid and given tube radii."""
    parameters_upscaled = []; parameters_extended = []
    for i, radius in enumerate(radii):
        parameters_upscaled.append( _load_parameters(fluid, radius, "ups") )
        parameters_extended.append( _load_parameters(fluid, radius, "ext") if _fit_extended[fluid] else {})
    return parameters_upscaled, parameters_extended


def save_fit_solutions(fluid, radii, parameters, times):
    """Saves the solution to the extended model of the `fluid` with the given `parameters` at the given `times`.
       The file scheme is `data/<fluid>_R<radius>mm_ext.csv`.
    """
    for i, radius in enumerate(radii):
        data = pandas.DataFrame({"t" : times[i]})
        for key, params in parameters[i].items():
            for j, p in enumerate(params):
                data["h_" + key + str(j)], data["v_" + key + str(j)] = _extended_model_integrator(p, data["t"])
                data["ca_" + key + str(j)] = _theta(p, data["v_" + key + str(j)])
        with open(os.path.join("data", fluid + "_R" + str(radius) + "mm_ext.csv"), 'w') as f:
            data.to_csv(f, index=False)


def _read_data(filenames):
    """Reads the CSV data from the given files and returns the combined DataFrame.

       The returned DataFrame contains the following columns:
           * "time" [s]
           * "rise" height [cm]
           * "CA" contact angle (measured in the rising fluid) [deg]
           * "theta" contact angle (measured in the receding fluid 2) [rad]
    """
    df = []
    for i,f in enumerate(filenames):
        df.append(pandas.read_csv(f, comment='#'))
        df[i]["file"] = i
    df = pandas.concat(df, ignore_index=True)
    df.sort_values(by="time", inplace=True)
    df.drop_duplicates(inplace=True) # remove multiple zeros
    df["theta"] = numpy.pi - numpy.deg2rad(df["CA"]) # Contact angle in fluid 1 given in degrees
    return df


def _load_parameters(fluid, radius, model):
    """Load the fitted parameters."""
    parameters = {s : [] for s in residual_types}
    for key, lst in parameters.items():
        for j in range(2):
            with open(os.path.join("data", fluid + "_R" + str(radius) + "mm_" + model + "_" + key + str(j) + ".param"), 'r') as fp:
                p = lmfit.Parameters()
                p.load(fp)
                lst.append(p)
    return parameters


def _write_fits_parameters(fp, fits, fluid, radius, model):
    """Write fit data into fp and save the parameters."""
    for key, res in fits.items():
        name = "height" if key == "h" else "theta" if key == 'ca' else "height & theta"
        fp.write("\n[" + model + ". model: " + name + " fit]\n")
        for j, r in enumerate(res):
            fp.write(lmfit.fit_report(r)+"\n")
            with open(os.path.join("data", fluid + "_R" + str(radius) + "mm_" + model + "_" + key + str(j) + ".param"), 'w') as fp2:
                r.params.dump(fp2)


def _write_fit_summary(filename, radii, fits):
    """Write fit summary (radius, eta, SD, reduced residual) into the file `filename`."""
    with open(filename, 'w') as f:
        f.write("R")
        for s in residual_types:
            f.write(",eta-" + s + ",eta-sd-" + s + ",slip-" + s + ",slip-sd-" + s + ",res-" + s)
        f.write("\n")
        for i, radius in enumerate(radii):
            for j in range(2):
                f.write(str(radius))
                for s in residual_types:
                    res = fits[i][s][j]
                    f.write("," + str(res.params["eta"].value) + "," + str(res.params["eta"].stderr) + ","
                            + str(res.params["slip"].value) + "," + str(res.params["slip"].stderr) + "," + str(res.redchi))
                f.write("\n")


################################################################################
#   MISCELLANEOUS
################################################################################

def _ordered_array(array):
    """Helper function for ordering an array in a revertible way."""
    if type(array) is not numpy.ndarray:
        array = array.to_numpy()
    order = array.argsort();
    order_rev = order.argsort()
    unique = array[order];
    mult = numpy.ones(numpy.size(order), dtype="int64")
    d = 0; i = 1
    while i < numpy.size(array):
        if(unique[i-d] == unique[i-d-1]):
            d += 1;
            unique = numpy.delete(unique, i-d)
            mult[i-d] += 1
            mult = numpy.delete(mult, numpy.size(unique))
        i += 1
    mult = numpy.repeat(range(numpy.size(mult)), mult)
    return (unique, mult, order_rev)


### Run the fitting for all fluids
if __name__ == "__main__":
    fit_and_save("glycerol")
    fit_and_save("soltrol", {"t_max": 400})
    fit_and_save("water")

