from psr_profile import *
from tseries import ParInputs, current_mjd
import numpy as np
import os, sys

# This is an admittedly clunky program whose main purpose is to quickly produce
# lots of dat files without the gui.  It plays no part in the functioning of
# the gui itself.

usg_str = "usage: python make_datfiles.py <ndatfiles> <min npsrs per file> <max npsrs per file> <tres> <length in sec> <basename> <output directory>"

if len(sys.argv) != 8:
    print usg_str
    sys.exit()

try:
    ndatfiles = int(sys.argv[1])
    minpsrs = int(sys.argv[2])
    maxpsrs = int(sys.argv[3])
    tres = float(sys.argv[4])
    length = float(sys.argv[5])
    basename = sys.argv[6]
    outpath = sys.argv[7]
except:
    print usg_str
    sys.exit()

def random_TS(output_dat, minpsrs, maxpsrs, tres, length, noise=100, **kwargs):
    """
    kwargs
    ______
    
    start_time: an MJD object, the time at which the time series begins.
        By default it's today's (integer) MJD.
    """
    start_time = kwargs.get('start_time', MJD(current_mjd().day_int))
    if output_dat[-4:] != '.dat': output_dat += '.dat'
    basename = output_dat.split('/')[-1][:-4]
    pathsplit = output_dat.split('/')[:-1]
    basedir = ''
    for word in pathsplit: basedir += (word + '/')
    pardir = '%s_parfiles' % (basedir + basename)
    if not os.path.exists(pardir): os.makedirs(pardir)

    # Generate parfiles and dat/inf files
    numpsrs = np.random.random_integers(minpsrs, maxpsrs)
    profiles = []
    profile_amps = []
    for ii in range(numpsrs):
        psr = ParInputs('Pulsar%02d'%(ii+1))
        parfile = create_parfile(psr.psr, float(psr.p0), float(psr.p1),\
            MJD(psr.posepoch), MJD(psr.pepoch), MJD(psr.t0),\
            float(psr.pb), float(psr.om), float(psr.e), float(psr.inc),\
            float(psr.m1), float(psr.m2), 'Amplitude: '+psr.amp)
        parfile_loc = '%s/%s_%s.par' % (pardir, basename, psr.psr)
        np.savetxt(parfile_loc, parfile, fmt='%s')
        profiles.append(psrProfile(parfile_loc))
        profile_amps.append(float(psr.amp))

    multi_psr_ts(profiles, profile_amps, start_time, tres, noise, length, basename, basedir)
    os.rename('%s.inf'%basename, '%s.inf'%(basedir+basename))

    for prof in profiles:
        prof.plot(tres=8.192e-7,\
            outfile=pardir+'/'+basename+'_'+prof.pars['PSR']+'.eps')

for ii in range(ndatfiles):
    fname = outpath + '/' + basename + '%02d'%(ii+1)
    random_TS(fname, minpsrs, maxpsrs, tres, length)
