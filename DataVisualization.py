import numpy
import matplotlib.pyplot

from DataAnalysis import *
from DataAnalysis import _theta


def add_all_data(radii, parameters, data, parameters_upscaled, parameters_extended, kws={}):
    """Adds the data from the fitting to the data frame list and returns names and colors for plotting."""
    names_ups = []; names_ext = []
    colors_ups = []; colors_ext = []
    for i, R_0 in enumerate(radii):
        data[i], nu, ne, cu, ce = _add_data(parameters[i], data[i], parameters_upscaled[i], parameters_extended[i], **kws)
        names_ups.append(nu); names_ext.append(ne)
        colors_ups.append(cu); colors_ext.append(ce)
    return data, names_ups, names_ext, colors_ups, colors_ext


def _add_data(parameters, data, parameters_upscaled, parameters_extended, t_max=numpy.inf):
    """Adds the data from the fitting to the data frame and returns names and
       colors for plotting. The final time `t_max` is used for solution of the
       extended model, which is continued by the stationary solution, if needed.
    """
    data_t = data["t"]; data_h = data["h"]; data_ca = data["theta"]

    # standard Lucas-Washborn solution
    data["h_LW"], data["ca_LW"] = height_theta(parameters, data_t, extended=False)
    data["h_resLW"] = data_h - data["h_LW"]
    data["ca_resLW"] = data_ca - data["ca_LW"]
    names_ups = ["LW"]; names_ext = ["LW"]
    colors_ups = ["#000000"]; colors_ext = ["#000000"]

    ### add models to the DataFrame
    for key, params in parameters_upscaled.items():
        for j, p in enumerate(params):
            name = "U_"+key+str(j)
            names_ups.append(name)
            colors_ups.append("#" + ("0000" if key == 'h' else "00" if key == 'ca' else "")
                                  + hex((j+1) * (255 // len(params))).lstrip("0x")
                                  + ("" if key == 'h' else "00" if key == 'ca' else "0000"))
            data["h_" + name], data["ca_" + name] = height_theta(p, data_t, extended=False)
            data["h_res" + name] = data_h - data["h_" + name]
            data["ca_res" + name] = data_ca - data["ca_" + name]

    mask = data_t.to_numpy() < t_max
    if not numpy.all(mask): print("Maximal time T = {}, {} data points disregarded.".format(t_max, numpy.count_nonzero(numpy.logical_not(mask))))
    for key, params in parameters_extended.items():
        for j, p in enumerate(params):
            name = "E_"+key+str(j)
            names_ext.append(name)
            colors_ext.append("#" + ("0000" if key == 'h' else "00" if key == 'ca' else "")
                                  + hex((j+1) * (255 // len(params))).lstrip("0x")
                                  + ("" if key == 'h' else "00" if key == 'ca' else "0000"))
            data["h_" + name] = 1.0
            data["ca_" + name] = _theta(p, 0.0)
            data.loc[mask, "h_" + name], data.loc[mask, "ca_" + name] = height_theta(p, data_t[mask], extended=True)
            data["h_res" + name] = data_h - data["h_" + name]
            data["ca_res" + name] = data_ca - data["ca_" + name]

    return data, names_ups, names_ext, colors_ups, colors_ext


def plot_results(radii, data, names, colors, figsize=[16,10]):
    """Plots the (height and contact angle) data and the fitted models given by names, as well as the residuals."""
    def forward(x):
        return numpy.sign(x)*abs(x)**(1/2)
    def inverse(x):
        return numpy.sign(x)*x**2

    for R_0, df, n, c in zip(radii, data, names, colors):
        ### plot models and residuals
        fig = matplotlib.pyplot.figure(figsize=figsize)
        ax = matplotlib.pyplot.subplot(2,2,1)
        df.plot.scatter(x="time", y="h", ax=ax, title="Capillary rise height over time for R = " + str(R_0) + "mm", label="data")
        df.plot.line(x="time", y=["h_" + s for s in n], color=c, ax=ax, linewidth=1, label=n)
        matplotlib.pyplot.xlim(left=0)
        ax.set_xscale('function', functions=(forward, inverse))
        ### plot residuals
        ax = matplotlib.pyplot.subplot(2,2,2)
        for s,col in zip(n[1:], c[1:]):
            df.plot(kind="scatter", x="time", y="h_res" + s, ax=ax, color=col, title="Residuals of height for R = " + str(R_0) + "mm", label=s)
        matplotlib.pyplot.xlim(left=0)
        ax.set_xscale('function', functions=(forward, inverse))

        ### plot cos(theta)
        ax = matplotlib.pyplot.subplot(2,2,3)
        ax = df.plot.scatter(x="time", y="theta", ax=ax, title="Contact angle over time for R = " + str(R_0) + "mm", label="data")
        ax = df.plot.line(x="time", y=["ca_" + s for s in n], color=c, ax=ax, linewidth=1, label=n)
        matplotlib.pyplot.xlim(left=0)
        ax.set_xscale('function', functions=(forward, inverse))
        ### plot residuals
        ax = matplotlib.pyplot.subplot(2,2,4)
        for s,col in zip(n[1:], c[1:]):
            df.plot(kind="scatter", x="time", y="ca_res" + s, ax=ax, color=col, title="Residuals of contact angle for R = " + str(R_0) + "mm", label=s)
        matplotlib.pyplot.xlim(left=0)
        ax.set_xscale('function', functions=(forward, inverse))


def analyze_eta(radii, parameters):
    """Analyzes and plots the fitted dyn. parameter eta."""
    N_model = 2
    eta_dim = numpy.zeros((N_model, len(radii), len(residual_types)))
    std_err_dim = numpy.zeros((N_model, len(radii), len(residual_types))) 

    for model in range(N_model):
        for i, R_0 in enumerate(radii):
            for j, typ in enumerate(residual_types):
                p = parameters[i][typ][model]
                eta_dim[model,i,j] = p["eta"] / p["U"]
                std_err_dim[model,i,j] = numpy.inf if p["eta"].stderr is None else p["eta"].stderr / p["U"]

    fig = matplotlib.pyplot.figure(figsize=[16,10])
    ax = matplotlib.pyplot.subplot(2,3,1, title="Parameter eta for varying radius")
    for m in range(N_model):
        for j, typ in enumerate(residual_types):
            matplotlib.pyplot.errorbar(x=radii, y=eta_dim[m,:,j], yerr=std_err_dim[m,:,j], label="M_"+str(m)+" "+typ)
    matplotlib.pyplot.xlabel("R [mm]")
    matplotlib.pyplot.xticks(ticks=radii)
    matplotlib.pyplot.legend()

    ax = matplotlib.pyplot.subplot(2,3,2, title="Parameter eta for varying fit type")
    for m in range(N_model):
        for i, R_0 in enumerate(radii):
            matplotlib.pyplot.errorbar(x=[j for j in range(len(residual_types))], y=eta_dim[m,i,:], yerr=std_err_dim[m,i,:], label="M_"+str(m)+" R="+str(R_0))
    matplotlib.pyplot.xlabel("Type")
    matplotlib.pyplot.xticks(ticks=[j for j in range(len(residual_types))], labels=residual_types)
    matplotlib.pyplot.legend()

    ax = matplotlib.pyplot.subplot(2,3,3, title="Parameter eta for varying model parameters")
    for j,typ in enumerate(residual_types):
        for i, R_0 in enumerate(radii):
            matplotlib.pyplot.errorbar(x=[m for m in range(N_model)], y=eta_dim[:,i,j], yerr=std_err_dim[:,i,j], label=typ+" R="+str(R_0))
    matplotlib.pyplot.xlabel("with slip")
    matplotlib.pyplot.xticks(ticks=[m for m in range(N_model)])
    matplotlib.pyplot.legend()

