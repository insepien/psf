import galsim
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as anim
from astropy.utils.console import ProgressBar
import warnings
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def set_alts_weights(nlayers):
    """
    set altitudes of each atmosphere layer and its weight on the final psf
    data is generated based on measurements from SCIDAR
    """
    Ellerbroek_alts = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    Ellerbroek_weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    Ellerbroek_interp = galsim.LookupTable(Ellerbroek_alts, Ellerbroek_weights, interpolant='linear')
    # create altitude with chosen num of layers
    alts = np.max(Ellerbroek_alts) * np.arange(nlayers) / (nlayers - 1)
    # find corresponding weights from table
    weights = Ellerbroek_interp(alts)  # interpolate the weights
    weights /= sum(weights)  # and renormalize
    return alts, weights


def set_wind_spd_dir(nlayers, max_speed, argr0_500, weights, seed):
    """
    setting the wind speed and direction for each atmosphere layer
    """
    # initiate random num
    rng = galsim.BaseDeviate(seed)
    u = galsim.UniformDeviate(rng)
    spd = []  # Wind speed in m/s
    dirn = []  # Wind direction in radians
    r0_500 = []  # Fried parameter in m at a wavelength of 500 nm
    for i in range(nlayers):
        spd.append(u() * max_speed)
        dirn.append(u() * 360 * galsim.degrees)
        r0_500.append(argr0_500 * weights[i] ** (-3. / 5))
        """
        print("Adding layer at altitude {:5.2f} km with velocity ({:5.2f}, {:5.2f}) m/s, "
              "and r0_500 {:5.3f} m."
              .format(alts[i], spd[i]*dirn[i].cos(), spd[i]*dirn[i].sin(), r0_500[i]))
        """
    return spd, dirn, r0_500


def fig_config(fig, psf_inst_ax, psf_intg_ax, psf_inst_axR, psf_vmax, psf_nx, psf_nxR, psf_scale, psf_scaleR, wf_vmax, coor, aper, x, y):
    # Axis for the inst PSF
    #fig.set_facecolor("k")
    color = sns.light_palette("seagreen", as_cmap=True)
    psf_inst_ax.set_xlabel("Arcsec")
    psf_inst_ax.set_ylabel("Arcsec")
    psf_inst_im = psf_inst_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                                     vmin=0.0, vmax=psf_vmax, cmap=color,
                                     extent=coor * 0.5 * psf_nx * psf_scale)
    psf_inst_ax.set_title("Instantaneous PSF")

    # Axis for the intg PSF
    psf_intg_ax.set_xlabel("Arcsec")
    #psf_intg_ax.set_ylabel("Arcsec")
    psf_intg_im = psf_intg_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                                     vmin=0.0, vmax=psf_vmax, cmap=color,
                                     extent=coor * 0.5 * psf_nx * psf_scale)
    psf_intg_ax.set_title("PSF integrated over time")

    # Axis for the wavefront image on the right.
    psf_inst_axR.set_xlabel("Arcsec")
    #psf_inst_axR.set_ylabel("Arcsec")
    psf_inst_imR = psf_inst_axR.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                                     vmin=0.0, vmax=psf_vmax, cmap=color,
                                     extent=coor * 0.5 * psf_nxR * psf_scaleR)
    psf_inst_axR.set_title("Instantaneous PSF for Rubin")
    
    etext_inst = psf_inst_ax.text(0.05, 0.92, '', transform=psf_inst_ax.transAxes,size=13)
    etext_intg = psf_intg_ax.text(0.05, 0.92, '', transform=psf_intg_ax.transAxes,size=13)
    etext_instR = psf_inst_axR.text(0.05, 0.92, '', transform=psf_inst_axR.transAxes,size=13)

    return psf_inst_im, psf_intg_im, psf_inst_imR, etext_inst, etext_intg, etext_instR


def create_GSO(t0, theta, lam, aper, time_step, atm):
    """
    create GSO objects for wavefront and PSF
    """
    # wavefront and psf
    psfinst = atm.makePSF(lam=lam, theta=theta, aper=aper,
                          t0=t0, exptime=time_step)
    psfintg = atm.makePSF(lam=lam, theta=theta, aper=aper,
                          t0=t0, exptime=time_step)
    psfinstR = atm.makePSF(lam=lam, theta=theta, aper=aper,
                          t0=t0, exptime=time_step)

    return psfinst, psfintg, psfinstR


def update_img(psf_img_sum, PSF, accumulate, i, psf_nx, psf_scale):
    """
    update images for each time step
    """
    psf_img0 = PSF.drawImage(nx=psf_nx, ny=psf_nx, scale=psf_scale)
    if accumulate:
        psf_img_sum += psf_img0
        psf_img_f = psf_img_sum / (i + 1)
    else:
        psf_img_f = psf_img0
    # Calculate simple estimate of size and ellipticity
    e = galsim.utilities.unweighted_shape(psf_img_f)
    return psf_img_f, e


def make_movie(args):
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 15
    rng = galsim.BaseDeviate(args.seed)

    # set altitude of each screen and its weight
    alts, weights = set_alts_weights(args.nlayers)

    # set wind speed and dir for each screen
    spd, dirn, r0_500 = set_wind_spd_dir(args.nlayers, args.max_speed, args.r0_500, weights, args.seed)

    # field angle at which to compute psf
    theta1 = (args.xs[0] * galsim.arcmin, args.ys[0] * galsim.arcmin)
    theta2 = (args.xs[1] * galsim.arcmin, args.ys[1] * galsim.arcmin)

    # atmosphere layers
    atm = galsim.Atmosphere(r0_500=r0_500, speed=spd, direction=dirn, altitude=alts, rng=rng,
                            screen_size=args.screen_size, screen_scale=args.screen_scale)
    # aperture of pupil
    aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                           nstruts=args.nstruts, strut_thick=args.strut_thick,
                           strut_angle=args.strut_angle * galsim.degrees,
                           screen_list=atm, pad_factor=args.pad_factor,
                           oversampling=args.oversampling)

    # create Fig frame
    metadata = dict(title='Wavefront Movie', artist='Matplotlib')
    writer = anim.FFMpegWriter(fps=15, bitrate=5000, metadata=metadata)
    fig, (psf_inst_ax1, psf_intg_ax1, psf_inst_axR) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),sharey=True, sharex=True)
    FigureCanvasAgg(fig)

    # configuring figures
    psf_inst_im1, psf_intg_im1, psf_inst_im1R, etext_inst1, etext_intg1, etext_inst1R = fig_config(fig, 
                                                                               psf_inst_ax1, psf_intg_ax1, psf_inst_axR, 
                                                                               args.psf_vmax, args.psf_nx,args.psf_nxR, args.psf_scale,
                                                                              args.psf_scaleR, args.wf_vmax, args.coord[0], aper,
                                                                              args.xs[0], args.ys[0])

    nstep = int(args.exptime / args.time_step)
    t0 = 0.0

    # store final image
    psf_inst_img_sum1 = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)
    psf_intg_img_sum1 = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)
    psf_inst_img_sum1R = galsim.ImageD(args.psf_nx, args.psf_nxR, scale=args.psf_scaleR)
    
    with ProgressBar(nstep) as bar:
        with writer.saving(fig, args.outfile, 100):
            for i in range(nstep):
                # create GSobjects
                psfinst1, psfintg1, psfinst1R = create_GSO(t0, theta1, args.lam, aper, args.time_step, atm)

                # draw and update images
                psf_inst_img_f1, e_inst1 = update_img(psf_inst_img_sum1, psfinst1, args.accumulate, i,  args.psf_nx,  args.psf_scale)
                psf_intg_img_f1, e_intg1 = update_img(psf_intg_img_sum1, psfintg1, args.accumulateint, i,  args.psf_nx,  args.psf_scale)
                psf_inst_img_f1R, e_inst1R = update_img(psf_inst_img_sum1R, psfinst1R, args.accumulate, i,  args.psf_nxR,  args.psf_scaleR)
                
                # Update t0 for the next movie frame.
                t0 +=  args.time_step

                # Matplotlib code updating plot elements
                psf_inst_im1.set_array(psf_inst_img_f1.array) 
                psf_intg_im1.set_array(psf_intg_img_f1.array)
                psf_inst_im1R.set_array(psf_inst_img_f1R.array)
                psf_inst_im1R.set_clim(vmin=0, vmax=np.max(psf_inst_img_f1R.array))
                
                etext_inst1.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_inst1['e1'], e_inst1['e2'], e_inst1['rsqr'] *  args.psf_scale ** 2))
                etext_intg1.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_intg1['e1'], e_intg1['e2'], e_intg1['rsqr'] *  args.psf_scale ** 2))
                etext_inst1R.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_inst1R['e1'], e_inst1R['e2'], e_inst1R['rsqr'] *  args.psf_scaleR ** 2)) 

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())
                bar.update()

if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
"""
Script to visualize the build up of an atmospheric PSF due to a frozen-flow Kolmogorov atmospheric
phase screens.  Note that the ffmpeg command line tool is required to run this script. Added integrated frame and shift 
location on focal plane 
"""), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--seed", type=int, default=1,
                        help="Random number seed for generating turbulence.  Default: 1")
    parser.add_argument("--r0_500", type=float, default=0.2,
                        help="Fried parameter at wavelength 500 nm in meters.  Default: 0.2")
    parser.add_argument("--nlayers", type=int, default=6,
                        help="Number of atmospheric layers.  Default: 6")
    parser.add_argument("--time_step", type=float, default=0.03,
                        help="Incremental time step for advancing phase screens and accumulating "
                             "instantaneous PSFs in seconds.  Default: 0.03")
    parser.add_argument("--exptime", type=float, default=3.0,
                        help="Total amount of time to integrate in seconds.  Default: 3.0")
    parser.add_argument("--screen_size", type=float, default=102.4,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 102.4")
    parser.add_argument("--screen_scale", type=float, default=0.1,
                        help="Resolution of atmospheric screen in meters.  Default: 0.1")
    parser.add_argument("--max_speed", type=float, default=20.0,
                        help="Maximum wind speed in m/s.  Default: 20.0")
    parser.add_argument( "--xs", type=list, default=[0,0],
                        help="x-coordinate of PSF in arcmin.  Default: 0.0")
    parser.add_argument( "--ys", type=list, default=[0,0],
                        help="y-coordinate of PSF in arcmin.  Default: 0.0")
    parser.add_argument("--lam", type=float, default=700.0,
                        help="Wavelength in nanometers.  Default: 700.0")
    parser.add_argument("--diam", type=float, default=4.0,
                        help="Size of circular telescope pupil in meters.  Default: 4.0")
    parser.add_argument("--obscuration", type=float, default=0.0,
                        help="Linear fractional obscuration of telescope pupil.  Default: 0.0")
    parser.add_argument("--nstruts", type=int, default=0,
                        help="Number of struts supporting secondary obscuration.  Default: 0")
    parser.add_argument("--strut_thick", type=float, default=0.05,
                        help="Thickness of struts as fraction of aperture diameter.  Default: 0.05")
    parser.add_argument("--strut_angle", type=float, default=0.0,
                        help="Starting angle of first strut in degrees.  Default: 0.0")
    parser.add_argument("--psf_nx", type=int, default=512,
                        help="Output PSF image dimensions in pixels.  Default: 512")
    parser.add_argument("--psf_nxR", type=int, default=13,
                       help="output psf size in pixels for rubin para")
    parser.add_argument("--psf_scale", type=float, default=0.005,
                        help="Scale of PSF output pixels in arcseconds.  Default: 0.005")
    parser.add_argument("--psf_scaleR", type=float, default=0.2,
                        help="psf scale for rubin")
    parser.add_argument("--accumulate", action='store_true',
                        help="Set to accumulate flux over exposure, as opposed to displaying the "
                             "instantaneous PSF.  Default: False")
    parser.add_argument("--accumulateint", action='store_false',
                        help="Set to accumulate flux over exposure, as opposed to displaying the "
                             "instantaneous PSF.  Default: True")

    parser.add_argument("--pad_factor", type=float, default=1.0,
                        help="Factor by which to pad PSF InterpolatedImage to avoid aliasing. "
                             "Default: 1.0")
    parser.add_argument("--oversampling", type=float, default=1.0,
                        help="Factor by which to oversample the PSF InterpolatedImage. "
                             "Default: 1.0")

    parser.add_argument("--psf_vmax", type=float, default=0.0003,
                        help="Matplotlib imshow vmax kwarg for PSF image.  Sets value that "
                             "maxes out the colorbar range.  Default: 0.0003")
    parser.add_argument("--wf_vmax", type=float, default=50.0,
                        help="Matplotlib imshow vmax kwarg for wavefront image.  Sets value "
                             "that maxes out the colorbar range.  Default: 50.0")

    parser.add_argument("--outfile", type=str, default="movie.mp4",
                        help="Output filename.  Default:movie.mp4")

    parser.add_argument("--coord", type=list, default=[np.r_[-1,1,-1,1],np.r_[-1,0,-1,0]],
                        help="Position of PSF center on focal plane. Default: (0',0'), and (-0.5',-0.5') ")

    args = parser.parse_args()
    make_movie(args)
