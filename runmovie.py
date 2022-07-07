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


def create_GSO(t0, theta, lam, aper, time_step, atm):
    """
    create GSO objects for wavefront and PSF
    """
    # wavefront and psf
    wf = atm.wavefront(aper.u, aper.v, t0, theta=theta) * 2 * np.pi / lam  # radians
    psfinst = atm.makePSF(lam=lam, theta=theta, aper=aper,
                          t0=t0, exptime=time_step)
    psfintg = atm.makePSF(lam=lam, theta=theta, aper=aper,
                          t0=t0, exptime=time_step)

    return wf, psfinst, psfintg


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


def fig_config(fig, psf_inst_ax, psf_intg_ax, wf_ax, psf_vmax, psf_nx, psf_scale, wf_vmax, coor, aper, xi, yi):
    # Axis for the inst PSF
    fig.set_facecolor("k")
    psf_inst_ax.set_xlabel("Arcsec")
    psf_inst_ax.set_ylabel("Arcsec")
    psf_inst_im = psf_inst_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                                     vmin=0.0, vmax=psf_vmax, cmap='hot',
                                     extent=coor * 0.5 * psf_nx * psf_scale)
    psf_inst_ax.set_title("PSF at (%d', %d') on focal plane " % (xi, yi))

    # Axis for the intg PSF
    psf_intg_ax.set_xlabel("Arcsec")
    psf_intg_ax.set_ylabel("Arcsec")
    psf_intg_im = psf_intg_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                                     vmin=0.0, vmax=psf_vmax, cmap='hot',
                                     extent=coor * 0.5 * psf_nx * psf_scale)
    psf_intg_ax.set_title("Integrated PSF")

    # Axis for the wavefront image on the right.
    wf_ax.set_xlabel("Meters")
    wf_ax.set_ylabel("Meters")
    wf_im = wf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                         vmin=-wf_vmax, vmax=wf_vmax, cmap='YlGnBu',
                         extent=np.r_[-1, 1, -1, 1] * 0.5 * aper.pupil_plane_size)

    divider = make_axes_locatable(wf_ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(wf_im, cax=cax, orientation='vertical')

    # Overlay an alpha-mask on the wavefront image showing which parts are actually illuminated.
    ilum = np.ma.masked_greater(aper.illuminated, 0.5)
    wf_ax.imshow(ilum, alpha=0.4, extent=np.r_[-1, 1, -1, 1] * 0.5 * aper.pupil_plane_size)

    # Color items white to show up on black background
    for ax in [psf_inst_ax, wf_ax, psf_intg_ax]:
        for _, spine in ax.spines.items():
            spine.set_color('w')
        ax.title.set_color('w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='both', colors='w')

    etext_inst = psf_inst_ax.text(0.05, 0.92, '', transform=psf_inst_ax.transAxes)
    etext_inst.set_color('w')

    etext_intg = psf_intg_ax.text(0.05, 0.92, '', transform=psf_intg_ax.transAxes)
    etext_intg.set_color('w')

    return psf_inst_im, psf_intg_im, wf_im, etext_inst, etext_intg


def make_movie(args):
    # set altitude of each screen and its weight
    alts, weights = set_alts_weights(args.nlayers)

    # set wind speed and dir for each screen
    spd, dirn, r0_500 = set_wind_spd_dir(args.nlayers, args.max_speed, args.argr0_500, args.weights, args.seed)

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
    fig, ([psf_inst_ax1, wf_ax1, psf_intg_ax1], [psf_inst_ax2, wf_ax2, psf_intg_ax2]) = plt.subplots(nrows=2, ncols=3,
                                                                                                     figsize=(15, 10))
    FigureCanvasAgg(fig)

    # configuring figures
    psf_inst_im1, psf_intg_im1, wf_im1, etext_inst1, etext_intg1 = fig_config(fig, psf_inst_ax1, psf_intg_ax1, wf_ax1,
                                                                              psf_vmax, psf_nx, psf_scale, wf_vmax,
                                                                              args.coord[0], aper, x1, y1)
    psf_inst_im2, psf_intg_im2, wf_im2, etext_inst2, etext_intg2 = fig_config(fig, psf_inst_ax2, psf_intg_ax2, wf_ax2,
                                                                              psf_vmax, psf_nx, psf_scale, wf_vmax,
                                                                              args.coord[1], aper, x2, y2)

    nstep = int(args.exptime / args.time_step)
    t0 = 0.0

    # store final image
    psf_inst_img_sum1 = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)
    psf_intg_img_sum1 = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)
    psf_inst_img_sum2 = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)
    psf_intg_img_sum2 = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)

    with ProgressBar(nstep) as bar:
        with writer.saving(fig, args.outfile, 100):
            for i in range(nstep):
                # create GSobjects
                wf1, psfinst1, psfintg1 = create_GSO(t0, theta1, args.lam, aper, time_step, atm)
                wf2, psfinst2, psfintg2 = create_GSO(t0, theta2, args.lam, aper, time_step, atm)

                # draw and update images
                psf_inst_img_f1, e_inst1 = update_img(psf_inst_img_sum1, psfinst1, args.accumulate, i, args.psf_nx,
                                                      args.psf_scale)
                psf_intg_img_f1, e_intg1 = update_img(psf_intg_img_sum1, psfintg1, args.accumulateint, i, args.psf_nx,
                                                      args.psf_scale)
                psf_inst_img_f2, e_inst2 = update_img(psf_inst_img_sum2, psfinst2, args.accumulate, i, args.psf_nx,
                                                      args.psf_scale)
                psf_intg_img_f2, e_intg2 = update_img(psf_intg_img_sum2, psfintg2, args.accumulateint, i, args.psf_nx,
                                                      args.psf_scale)

                # Update t0 for the next movie frame.
                t0 += args.time_step

                # Matplotlib code updating plot elements
                wf_im1.set_array(wf1)
                psf_inst_im1.set_array(psf_inst_img_f1.array)
                psf_intg_im1.set_array(psf_intg_img_f1.array)
                wf_ax1.set_title("Wavefront Image. t={:5.2f} s".format(i * args.time_step))
                etext_inst1.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_inst1['e1'], e_inst1['e2'], e_inst1['rsqr'] * args.psf_scale ** 2))
                etext_intg1.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_intg1['e1'], e_intg1['e2'], e_intg1['rsqr'] * args.psf_scale ** 2))

                wf_im2.set_array(wf2)
                psf_inst_im2.set_array(psf_inst_img_f2.array)
                psf_intg_im2.set_array(psf_intg_img_f2.array)
                wf_ax2.set_title("Wavefront Image. t={:5.2f} s".format(i * args.time_step))
                etext_inst2.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_inst2['e1'], e_inst2['e2'], e_inst2['rsqr'] * args.psf_scale ** 2))
                etext_intg2.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                    e_intg2['e1'], e_intg2['e2'], e_intg2['rsqr'] * args.psf_scale ** 2))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())
                bar.update()


if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(description=(
        """
        Script to visualize the build up of an atmospheric PSF due to a frozen-flow Kolmogorov atmospheric
        phase screens.  Note that the ffmpeg command line tool is required to run this script.
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
    parser.add_argument("-xs", "--x", type=list, default=[0, 0],
                        help="x-coordinate of PSF in arcmin.  Default: 0.0")
    parser.add_argument("-ys", "--y", type=list, default=[0, 0],
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
    parser.add_argument("--psf_scale", type=float, default=0.005,
                        help="Scale of PSF output pixels in arcseconds.  Default: 0.005")
    parser.add_argument("--accumulate", action='store_true',
                        help="Set to accumulate flux over exposure, as opposed to displaying the "
                             "instantaneous PSF.  Default: False")

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

    parser.add_argument("--outfile", type=str, default="output/psf_wf_movie.mp4",
                        help="Output filename.  Default: output/psf_wf_movie.mp4")

    parser.add_argument("--output_coordinate", type=list, default=[np.r_[-1, 1, -1, 1], np.r_[-1, 0, -1, 0]],
                        help="Position of the PSF center on output image. Default: (0,0) top, (-0.5,-0.5) bottom")

    args = parser.parse_args()
    make_movie(args)
