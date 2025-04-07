import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import sunpy.visualization.colormaps
from astropy.coordinates import SkyCoord
from sunpy.coordinates.sun import carrington_rotation_time
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm

# os.nice(15)
outdir = '/Users/clowder/data/polar/'
datadir = '/Users/clowder/data/aia_rotation/'
# outdir = '/sol/d0/lowder/polar/'
# datadir = '/sol/d0/lowder/aia/'

def keynote_figs():
    matplotlib.rcParams['lines.color'] = 'white'
    matplotlib.rcParams['patch.edgecolor'] = 'white'
    matplotlib.rcParams['text.color'] = 'white'
    matplotlib.rcParams['axes.facecolor'] = 'black'
    matplotlib.rcParams['axes.edgecolor'] = 'white'
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    matplotlib.rcParams['xtick.color'] = 'white'
    matplotlib.rcParams['ytick.color'] = 'white'
    matplotlib.rcParams['grid.color'] = 'white'
    matplotlib.rcParams['figure.facecolor'] = 'black'
    matplotlib.rcParams['figure.edgecolor'] = 'black'
    matplotlib.rcParams['savefig.facecolor'] = 'black'
    matplotlib.rcParams['savefig.edgecolor'] = 'black'
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['text.usetex'] = False

keynote_figs()

def generate_polarmap(cr:int, 
                      wave:int = 193, 
                      datadir:str = '/Users/clowder/data/aia_rotation/') -> None:
    """Generate polar map from input EUV data."""
    t0 = carrington_rotation_time(cr)
    t1 = carrington_rotation_time(cr+1)

    result = Fido.search(a.Time(t0, t1), a.Instrument.aia, a.Wavelength(wave * u.angstrom), a.Sample(1*u.day))

    files = Fido.fetch(result, path=datadir)

    out_shape=(4096,4096)
    block_size=(1024,1024)

    data_np = np.full(out_shape, np.nan)
    data_sp = np.full(out_shape, np.nan)

    for file in files:
        aia_map = sunpy.map.Map(file)
        aia_map.data[...] = aia_map.data[...] / aia_map.meta['EXPTIME']

        obs_np = SkyCoord(0*u.deg, 90*u.deg, 1*u.AU, obstime=aia_map.date, frame='heliographic_carrington', observer='self')
        out_ref_coord_np = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=obs_np.obstime,
                                frame='helioprojective', observer=obs_np,
                                rsun=aia_map.coordinate_frame.rsun)
        out_header_np = sunpy.map.make_fitswcs_header(
            out_shape,
            out_ref_coord_np,
            scale=u.Quantity(aia_map.scale),
            instrument=aia_map.instrument,
            wavelength=aia_map.wavelength
        )
        outmap_np = aia_map.reproject_to(out_header_np, algorithm='adaptive', block_size=block_size, parallel=True)

        obs_sp = SkyCoord(0*u.deg, -90*u.deg, 1*u.AU, obstime=aia_map.date, frame='heliographic_carrington', observer='self')
        out_ref_coord_sp = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=obs_sp.obstime,
                                frame='helioprojective', observer=obs_sp,
                                rsun=aia_map.coordinate_frame.rsun)
        out_header_sp = sunpy.map.make_fitswcs_header(
            out_shape,
            out_ref_coord_sp,
            scale=u.Quantity(aia_map.scale),
            instrument=aia_map.instrument,
            wavelength=aia_map.wavelength
        )
        outmap_sp = aia_map.reproject_to(out_header_sp, algorithm='adaptive', block_size=block_size, parallel=True)

        data_np = np.fmin(data_np, outmap_np.data*(outmap_np.data!=0))
        data_sp = np.fmin(data_sp, outmap_sp.data*(outmap_sp.data!=0))
    
    cmap = f"sdoaia{aia_map.wavelength.value.astype(int)}"
    wavelength = aia_map.wavelength.value.astype(int)

    map_np = sunpy.map.Map(data_np, out_header_np)
    map_sp = sunpy.map.Map(data_sp, out_header_sp)

    map_np.save(outdir + f"cr{cr}-np-{wavelength}A.fits")
    map_sp.save(outdir + f"cr{cr}-sp-{wavelength}A.fits")

    fig, ax = plt.subplots(figsize=(9.5, 7.5), subplot_kw={'projection':outmap_np.wcs})

    im = ax.imshow(data_np, cmap=matplotlib.colormaps[cmap], vmin=0, vmax=140)

    ax.set_facecolor('black')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"North Pole - SDO/AIA - {wavelength}$\\AA$ - CR {cr}")
    fig.colorbar(im, ax=ax, label='min(DN/s)', extend='max')

    plt.savefig(f"./plt/cr{cr}-np-{wavelength}A.pdf")
    plt.savefig(f"./plt/cr{cr}-np-{wavelength}A.png", dpi=300)

    plt.close()

    fig, ax = plt.subplots(figsize=(9.5, 7.5), subplot_kw={'projection':outmap_sp.wcs})

    im = ax.imshow(data_sp, cmap=matplotlib.colormaps[cmap], vmin=0, vmax=140)

    ax.set_facecolor('black')
    # ax.coords.grid(color='white', alpha=.25, ls='dotted')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"South Pole - SDO/AIA - 193$\\AA$ - CR {cr}")
    fig.colorbar(im, ax=ax, label='min(DN/s)', extend='max')

    plt.savefig(f"./plt/cr{cr}-sp-{wavelength}A.pdf")
    plt.savefig(f"./plt/cr{cr}-sp-{wavelength}A.png", dpi=300)

    plt.close()


# crs = np.arange(2097,2284)
crs = np.arange(2098,2099)
for cr in tqdm(crs):
    generate_polarmap(cr)


input_pattern = './plt/aia_*.png'
output_file = './mov/aia_cycle.mp4'

output_width = 2048
output_height = 2048


# ffmpeg -framerate 8 -pattern_type glob -i './plt/cr*-np-193A.png' -vcodec libx264 -vf format=yuv420p ./mov/polar-np.mp4
# ffmpeg -framerate 8 -pattern_type glob -i './plt/cr*-sp-193A.png' -vcodec libx264 -vf format=yuv420p ./mov/polar-sp.mp4