from psr_profile import psrProfile, MJD, create_parfile, multi_psr_ts
from tseries import ParInputs, current_mjd
import numpy as np
import os, sys

# This is an admittedly clunky program whose main purpose is to quickly produce
# lots of dat files without the gui.  It plays no part in the functioning of
# the gui itself.

usg_str = "usage: python make_datfiles.py <nstart> <nend> <min npsrs per file> <max npsrs per file> <tres> <length in sec> <basename> <output directory>"

if len(sys.argv) != 9:
    print usg_str
    sys.exit()

try:
    nstart = int(sys.argv[1])
    nend = int(sys.argv[2])
    minpsrs = int(sys.argv[3])
    maxpsrs = int(sys.argv[4])
    tres = float(sys.argv[5])
    length = float(sys.argv[6])
    basename = sys.argv[7]
    outpath = sys.argv[8]
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
    current_dir = os.getcwd()

    start_time = kwargs.get('start_time', MJD(current_mjd().day_int))
    if output_dat[-4:] != '.dat': output_dat += '.dat'
    basename = output_dat.split('/')[-1][:-4]
    pathsplit = output_dat.split('/')[:-1]
    basedir = ''
    for word in pathsplit: basedir += (word + '/')
    pardir = '%s_parfiles' % (basedir + basename)
    if not os.path.exists(pardir): os.makedirs(pardir)

    os.chdir(basedir)

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
#    os.rename('%s.inf'%basename, '%s.inf'%(basedir+basename))

    for prof in profiles:
        prof.plot(tres=8.192e-7,\
            outfile=pardir+'/'+basename+'_'+prof.pars['PSR']+'.eps')

    os.chdir(current_dir)

for ii in range(nstart, nend+1):
    fname = outpath + '/' + basename + 'n' + '%02d'%ii
    random_TS(fname, minpsrs, maxpsrs, tres, length)
