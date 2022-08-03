import argparse
import treegp
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.gaussian_process.kernels import Kernel
import os
import treecorr

# plotting things
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12


# probably looks familiar :)
def comp_2pcf_treecorr(x, y, k, bin_type='TwoD'):
    """measure the anisotropic 2pcf of some data."""
    cat = treecorr.Catalog(x=x, y=y, k=k, w=None)
    kk = treecorr.KKCorrelation(min_sep=0, max_sep=0.3, nbins=17,
                                bin_type=bin_type, bin_slop=0)
    kk.process(cat)

    return kk

## really don't worry about what this is going!! 
def eval_kernel(kernel):
    """
    Some import trickery to get all subclasses 
    of sklearn.gaussian_process.kernels.Kernel
    into the local namespace without doing 
    "from sklearn.gaussian_process.kernels import *"
    and without importing them all manually.
    Example:
    kernel = eval_kernel("RBF(1)") instead of
    kernel = sklearn.gaussian_process.kernels.RBF(1)
    """
    def recurse_subclasses(cls):
        out = []
        for c in cls.__subclasses__():
            out.append(c)
            out.extend(recurse_subclasses(c))
        return out
    clses = recurse_subclasses(Kernel)
    for cls in clses:
        module = __import__(cls.__module__, globals(), locals(), cls)
        execstr = "{0} = module.{0}".format(cls.__name__)
        exec(execstr, globals(), locals())

    from numpy import array

    try:
        k = eval(kernel)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to evaluate kernel string {0!r}.  "
                           "Original exception: {1}".format(kernel, e))

    if isinstance(k.theta, property):
        raise TypeError("String provided was not initialized properly")
    return k

# or this :)
def get_correlation_length_matrix(size, g1, g2):
    if abs(g1) > 1:
        g1 = 0
    if abs(g2) > 1:
        g2 = 0
    g = np.sqrt(g1**2 + g2**2)
    q = (1-g) / (1+g)
    phi = 0.5 * np.arctan2(g2, g1)
    rot = np.array([[np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]])
    ell = np.array([[size**2, 0],
                    [0, (size * q)**2]])
    L = np.dot(rot.T, ell.dot(rot))
    return L

def plot_model(axes, psf_param, measured, pcf_fit, extent):
    """Plot the psf parameter info on the set of axes passed in."""
    cov_cmap = sns.color_palette("vlag", as_cmap=True)
    vmax, vmin = 2e-5, -2e-5

    im = axes[0].imshow(measured.xi, extent=[extent[0], extent[-1], 
                                             extent[0], extent[-1]], 
                        origin='lower', vmax=vmax, vmin=vmin, cmap=cov_cmap)
    axes[0].set_ylabel(r'$\Delta y$', labelpad=-2)

    divider = make_axes_locatable(axes[0])
    ax_cb = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=ax_cb)

    im = axes[1].imshow(pcf_fit, extent=[extent[0], extent[-1],
                                         extent[0], extent[-1]],
                        origin='lower', cmap=cov_cmap, vmax=vmax, vmin=vmin)
    divider = make_axes_locatable(axes[1])
    ax_cb = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=ax_cb)


    im = axes[2].imshow(measured.xi-pcf_fit, extent=[extent[0], extent[-1],
                                                     extent[0], extent[-1]],
                        origin='lower', cmap=cov_cmap, vmax=vmax, vmin=vmin)
    divider = make_axes_locatable(axes[2])
    ax_cb = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=ax_cb)

    return axes


def fit_2pcf(x, param):
    """Use treegp to get the best fit Komogorov kernel."""
    sigma = .01
    # these (size, g1, g1) are the kernel parameters, not the PSF parameters!!!
    size, g1, g2 = 0.15, .1, .1
    L = get_correlation_length_matrix(size, g1, g2)
    invL = np.linalg.inv(L)
    kernel = "%f" % (sigma**2) + " * AnisotropicVonKarman(invLam={0!r})".format(invL)
    kernel = eval_kernel(kernel)

    fitter = treegp.two_pcf(x, param, y_err=np.zeros(param.shape), min_sep=0.,
                            max_sep=.3, nbins=17,
                            anisotropic=True, p0=[10, 0., 0.])
    opt_kernel = fitter.optimizer(kernel)

    # visualize
    pixel_squareroot = 17
    npixels = pixel_squareroot**2
    x = np.linspace(-.3, .3, pixel_squareroot)
    x1, x2 = np.meshgrid(x, x)
    coord = np.array([x1.reshape(npixels), x2.reshape(npixels)]).T
    pcf = opt_kernel.__call__(coord, Y=np.zeros_like(coord))[:, 0]
    pcf = pcf.reshape((pixel_squareroot, pixel_squareroot))

    return pcf, x, opt_kernel


def get_second_moment_ellipticities(L):
    """Do some math to get ellipticity parameters from the kernel."""
    l11 = L[0, 0]
    l12 = L[0, 1]
    l22 = L[1, 1]
    detL = l11 * l22 - l12**2

    denom = l11 + l22 + 2 * np.sqrt(detL)
    e1 = (l11 - l22) / denom
    e2 = l12 / denom

    sigplus = np.sqrt((l11 + l22) / 2)
    sigminus = detL**(1/4)

    return {"sigp": sigplus, "sigm": sigminus, 'g1': e1, 'g2': e2}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simfile", type=str)
    parser.add_argument("--simdir", type=str)
    parser.add_argument("--dictfile", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plotfile", type=str)
    parser.add_argument("--plotdir", type=str)
    args = parser.parse_args()

    today = datetime.today().strftime('%m%d%y')
    
    # load dictfile if it already exists, and add a dict entry corresponding 
    # to this random seed. If it doesn't exist, define it.
    s = args.seed
    try:
        kplist_dict = pickle.load(open(args.dictfile, 'rb'))
        kplist_dict['seed'].append(s)
    except FileNotFoundError:
        kplist_dict = {psf_par: {'k': [], 'sigp': [], 'sigm': [],
                                 'g1': [], 'g2': [], 'amp':[], 'resid':[]}
                       for psf_par in ['e1', 'e2', 'sigma']}
        kplist_dict['seed'] = [s]

    # load data
    d = pickle.load(open(os.path.join(args.simdir, args.simfile), 'rb'))
    for k in ['e1', 'e2', 'thx', 'thy', 'sigma']:
        locals()[k] = np.array(d[k])
    d_sigma = sigma - np.mean(sigma)
    x = np.array([thx, thy]).T

    # start a figure
    f, (a_size, a_1, a_2) = plt.subplots(3, 3, figsize=(10,8), sharey=True, sharex=True)
    
    for psf_par, label, ax in zip([d_sigma, e1, e2],
                                  ['sigma', 'e1', 'e2'],
                                  [a_size, a_1, a_2]):

        # fitting here is done through a package written by P-F Léget called
        # treegp. it takes care of fitting the anisotropic 2pcf to the profile 
        # we talked about with Pat. It uses the terminology of "kernel" to
        # describe what we've talked about as a correlation function; so the
        # "kernel" that floats around in this code is actually the modeled/fit
        # to the data correlation function.
        measured = comp_2pcf_treecorr(thx, thy, psf_par)
        pcf, extent, kernel = fit_2pcf(x, psf_par)
        # save the kernel in the dict
        kplist_dict[label]['k'].append(kernel)
        
        # save different summary quantities of the residual to dict
        residual = measured.xi-pcf
        kplist_dict[label]['resid'].append({'sum': np.sum(residual),
                                            'sumofabs': np.sum(np.abs(residual)),
                                            'sumof2': np.sum(residual**2)})

        # this part extracts e1/e2 from the kernel above
        invLam = kernel.get_params()['k2'].invLam
        L = np.linalg.inv(invLam)
        kernel_params = get_second_moment_ellipticities(L)
        for kpar, kval in kernel_params.items():
            kplist_dict[label][kpar].append(kval)
        kplist_dict[label]['amp'].append(kernel.get_params()['k1'].constant_value)

        # plot a row of measured 2pcf, fit, and residual for the given psf parameter.
        ax = plot_model(ax, psf_par, measured, pcf, extent)
    
    # plotting things
    [a_size[i].set_title(['measured', 'model', 'residual'][i]) for i in range(3)]
    [a_2[i].set_xlabel(r'$\Delta x$') for i in range(3)]
    [a.text(-0.55, 0.3, p, fontsize=16) 
     for (a,p) in zip([a_size[0], a_1[0], a_2[0]],
                      [r'$\delta \sigma$', r'$e_1$', r'$e_2$'])]

    # save the figure 
    #details = blah 
    #plt.savefig(f'plots/{today}/2pcffit_{details}_{today}.png', dpi=150)
    # or:
    plt.savefig(os.path.join(args.plotdir,args.plotfile), dpi=150)
    # and put this line up above: parser.add_argument("--plotfile", type=str)
    plt.clf()

    # save the dictfile
    pickle.dump(kplist_dict, open(args.dictfile, 'wb'))
