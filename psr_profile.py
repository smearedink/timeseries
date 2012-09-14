import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import figure, xlabel, ylabel, xlim, ylim, plot, savefig, imshow, subplot, subplots_adjust, xticks, yticks
import matplotlib.cm as cm
import sys
import os
import struct
from time import time as current_time
from presto import read_inffile, writeinf, infodata
from numpy.polynomial.polynomial import polyval

TSUN = 4.925490947e-6
OMFAC = 5.53061466e-10

#   ____  ____  ____  ____  ____   __  ____  __  __    ____ 
#  (  _ \/ ___)(  _ \(  _ \(  _ \ /  \(  __)(  )(  )  (  __)
#   ) __/\___ \ )   / ) __/ )   /(  O )) _)  )( / (_/\ ) _) 
#  (__)  (____/(__\_)(__)  (__\_) \__/(__)  (__)\____/(____)

class psrProfile:
    """
    Generate a random pulsar profile from a cluster of Gaussian functions.  If
    a parfile is provided, pulsar and binary parameters may be used to
    calculate Doppler shifts, spin-down, etc.

    maxpeaks: the maximum number of Gaussians used to generate the profile
    mass: mass of the pulsar in solar mass units
    """
    def __init__(self, parfile=None, maxpeaks=5, mass=1.4):
        self.mass = mass
        
        self.parfile = parfile

        if parfile is None:
            self.pars = None
            self.p0 = 1.0
            self.p1 = 0.0
            self.f0 = 1.0
            self.f1 = 0.0
            self.dm = 0.0
            self.pb = 1000.0
        else: self.load_parfile(parfile)

        self.numpeaks = np.random.randint(1,maxpeaks+1)
        self.peaks = []
        nextamp = 1.0
        pp = self.numpeaks
        thickness = np.random.normal(1.0, 0.2)
        while np.abs(thickness-1.0) > 0.9:
            thickness = np.random.normal(1.0, 0.2)
        while pp > 0:
            sig = thickness*np.random.normal(0.01, 0.001)
            ph = thickness*(np.random.normal(0.5, 0.02)-0.5)+0.5
            self.peaks.append(np.array([nextamp, sig, ph]))
            nextamp *= np.square(np.random.rand())
            pp -= 1
        self.peaks = np.array(self.peaks)

        # rescale so that the whole profile has max height ~1.0
        if self.numpeaks > 1:
            heights = []
            for tr in [0.00008, 0.00009, 0.00010, 0.00011, 0.00012]:
                heights.append(np.max(self.bin_prof(tres=tr)))
            self.peaks = self.scale_height(1./np.max(heights))

    def load_parfile(self, parfile):
        """
        Load parfile into dictionary attribute 'pars' and make attributes for
        p0 and p1, and binary parameters asini, ecc, t0, pb, sini, and om.
        """
        self.pars = dict(np.genfromtxt(parfile, dtype=str, usecols=(0,1)))

        if self.pars.has_key('P0'):
            p0 = self.pars['P0']
            if self.pars.has_key('P1'): p1 = self.pars['P1']
            else: p1 = 0.
            self.p0 = float(p0.replace('D', 'E'))
            self.f0 = 1./self.p0
            self.p1 = float(p1.replace('D', 'E'))
            self.f1 = -self.p1/(self.p0*self.p0)
        else:
            f0 = self.pars['F0']
            if self.pars.has_key('F1'): f1 = self.pars['F1']
            else: f1 = 0.
            self.f0 = float(f0.replace('D', 'E'))
            self.p0 = 1./self.f0
            self.f1 = float(f1.replace('D', 'E'))
            self.p1 = -self.f1/(self.f0*self.f0)

        dm = self.pars['DM']
        self.dm = float(dm.replace('D', 'E'))

        pb = self.pars['PB']
        self.pb = float(pb.replace('D', 'E'))*86400.

    def phase(self, ph):
        """
        Return height of profile (normalized so that max height = 1) at given
        pulse phase.
        """
        ph -= np.floor(ph)
        val = 0.
        for peak in self.peaks:
            val += peak[0]*np.exp(-0.5*np.square((ph-peak[2])/peak[1]))
        return val
        
    def bin_prof(self, tres=8.192e-5, binshift=0.0):
        """
        Output the profile as a numpy array binned according to the provided
        time resolution (using the attribute p0 as the length of the profile).

        Almost certainly the profile will end partway through the last bin.
        This last bin is completed, but the profile is scaled and positioned as
        though it were cut off partway.

        binshift: (non-integer) number of bins to shift the peak.
        """
        nbins = self.p0/tres
        prof = np.zeros(np.ceil(nbins))
        for peak in self.peaks:
            prof += gaussian(nbins, peak[1]*nbins, peak[2]*nbins+binshift,\
                             peak[0], 0.0, nbins)
        return prof

    def plot(self, tres=8.192e-5, outfile='profile.eps'):
        """
        Output a plot of the profile.

        tres: time resolution in seconds
        outfile: output file for plot if desired--format deduced from extension
        """
        nbins = np.ceil(self.p0/tres)
        hiphase = nbins/(self.p0/tres)
        axis = np.linspace(0, hiphase, nbins, endpoint=False)
        figure()
        xlabel('Phase') 
        ylim(-0.1, 1.1)
        xlim(0., 1.)
        plot(axis, self.bin_prof(tres=tres))
        savefig(outfile, bbox_inches='tight')

    def scale_height(self, factor):
        """
        Return the set of peaks with height scaled by given factor.
        """
        scaled_peaks = self.peaks.copy()
        for ii in range(self.numpeaks): scaled_peaks[ii][0] *= factor
        return scaled_peaks

    def scale_width(self, factor):
        """
        Return the set of peaks with width scaled by given factor.
        """
        scaled_peaks = self.peaks.copy()
        for ii in range(self.numpeaks): scaled_peaks[ii][1] *= factor
        return scaled_peaks

#        ___            ___         ___     
#       /\__\          /\  \       /\  \    
#      /::|  |         \:\  \     /::\  \   
#     /:|:|  |     ___ /::\__\   /:/\:\  \  
#    /:/|:|__|__  /\  /:/\/__/  /:/  \:\__\ 
#   /:/ |::::\__\ \:\/:/  /    /:/__/ \:|__|
#   \/__/~~/:/  /  \::/  /     \:\  \ /:/  /
#         /:/  /    \/__/       \:\  /:/  / 
#        /:/  /                  \:\/:/  /  
#       /:/  /                    \::/__/   
#       \/__/                      ~~       

class MJD():
    """
    MJD class that contains the integer and fractional portions of the date
    separately to increase precision.
    """
    def __init__(self, day_int, day_frac=None):
        if day_frac is not None:
            self.day_int = int(day_int)
            self.day_frac = float(day_frac)
        elif type(day_int) is not str:
            if isinstance(day_int, MJD):
                self.day_int = day_int.day_int
                self.day_frac = day_int.day_frac
            else:
                self.day_int = int(day_int)
                self.day_frac = np.floor((day_int-np.floor(day_int))*1.e11)/\
                    (1.e11)
        else:
            val = eval(day_int)
            if type(val) is int:
                self.day_int = val
                self.day_frac = 0.0
            else:
                self.day_int = eval(day_int.split('.')[0])
                self.day_frac = eval('.'+day_int.split('.')[1])

    def add(self, time):
        if time < 0.0:
            return self.subtract(-time)
        elif self.day_frac + time < 1.0:
            return MJD(self.day_int, self.day_frac + time)
        else:
            int_add = int(time)
            frac_add = time - np.floor(time)
            return self.add_MJD(MJD(int_add, frac_add))

    def subtract(self, time):
        if time < 0.0:
            return self.add(-time)
        elif self.day_frac - time >= 0.0:
            return MJD(self.day_int, self.day_frac - time)
        else:
            int_sub = int(time)
            frac_sub = time - np.floor(time)
            return self.subtract_MJD(MJD(int_sub, frac_sub))

    def add_MJD(self, other_MJD):
        if self.day_frac + other_MJD.day_frac < 1.0:
            return MJD(self.day_int + other_MJD.day_int,\
                       self.day_frac + other_MJD.day_frac)
        else:
            return MJD(self.day_int + other_MJD.day_int + 1,\
                       self.day_frac + other_MJD.day_frac - 1.0)

    def __add__(self, days):
        if isinstance(days, MJD): return self.add_MJD(days)
        else: return self.add(days)

    def subtract_MJD(self, other_MJD):
        if self > other_MJD: 
            if self.day_frac - other_MJD.day_frac >= 0.0:
                return MJD(self.day_int - other_MJD.day_int,\
                    self.day_frac - other_MJD.day_frac)
            else:
                return MJD(self.day_int - other_MJD.day_int - 1,\
                    self.day_frac - other_MJD.day_frac + 1.0)
        else:
            int_part = float(self.day_int - other_MJD.day_int)
            frac_part = self.day_frac - other_MJD.day_frac
            return int_part + frac_part

    def __sub__(self, days):
        if isinstance(days, MJD): return self.subtract_MJD(days)
        else: return self.subtract(days)

    def __rsub__(self, days):
        return days + -self

    def __neg__(self):
        if self.day_int >= 1:
            return -float(self)

    def as_float(self):
        return self.day_int + self.day_frac

    def show(self):
        return repr(self.day_int)+repr(self.day_frac)[1:]

    def __repr__(self):
        return self.show()

    def __float__(self):
        return self.as_float()

    def __eq__(self, other):
        other = MJD(other)
        return (self.day_int == other.day_int) and\
            (self.day_frac == other.day_frac)

    def __ne__(self, other):
        other = MJD(other)
        return (self.day_int != other.day_int) or\
            (self.day_frac != other.day_frac)

    def __lt__(self, other):
        other = MJD(other)
        if (self.day_int == other.day_int):
            return self.day_frac < other.day_frac
        else:
            return self.day_int < other.day_int

    def __gt__(self, other):
        other = MJD(other)
        if (self.day_int == other.day_int):
            return self.day_frac > other.day_frac
        else:
            return self.day_int > other.day_int

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

#  ____________                ________           _____             
#  ___  __/__(_)______ __________  ___/______________(_)____________
#  __  /  __  /__  __ `__ \  _ \____ \_  _ \_  ___/_  /_  _ \_  ___/
#  _  /   _  / _  / / / / /  __/___/ //  __/  /   _  / /  __/(__  ) 
#  /_/    /_/  /_/ /_/ /_/\___//____/ \___//_/    /_/  \___//____/  

class TimeSeries():
    """
    Generate a time series with noise using a psrProfile object.

    THIS DESCRIPTION NEEDS EDITING

    profile: a psrProfile object with an associated parfile
    pulseheight: max height of pulses
    start: MJD at beginning of time series
    tres: time resolution in seconds
    noise: standard deviation of white noise
    length: duration of time series in seconds
    obs: observatory code (-1 is barycentred)
    nchunks: minimum number of chunks (might be increased if necessary)
    zero:
    """
    def __init__(self, profile, pulseheight=256., start=MJD(52000, 0.0),\
                 tres=8.192e-5, noise=1000.0, length=1000.0, obs=-1,\
                 nchunks=1, zero=0.):
 
        self.profile = profile
        self.start = start
        self.tres = tres
        self.noise = noise
        self.zero = zero
        self.pulseheight = pulseheight
        self.nbins = int(np.ceil(length/tres))
        if self.nbins % 2: self.nbins -= 1
        self.length = self.nbins*tres
        self.nchunks = nchunks

        # make sure there are at least 5 chunks per orbit
        chunklength = self.length/self.nchunks
        onefifthpb = profile.pb/5.
        if chunklength > onefifthpb:
            self.nchunks = int(np.ceil(self.length/onefifthpb))

        if obs is -1: self.bary = True
        else: self.bary = False

        if noise: self.ts_noise = np.random.normal(loc=zero, scale=noise,\
                                                   size=self.nbins)
        else: self.ts_noise = np.zeros(self.nbins) + zero

        self.ts_pulses = np.zeros(self.nbins)

        split_ts_pulses = np.array_split(self.ts_pulses, self.nchunks)

        print "Generating time series for pulsar %s..." %\
            self.profile.pars['PSR']

        time = start
        chunkcount = 0
        for chunk in split_ts_pulses:
            chunkcount += 1
            print "Making chunk %d of %d..." % (chunkcount, self.nchunks)
            chunklen = len(chunk)
            tobs = chunklen*tres
            if tobs % 60: tobs += 60. - (tobs % 60)
            pco = polyCo(profile, time, tobs, obs)
            phase_arr = pco.calc_phases(tres, chunklen)
            chunk += pulseheight*profile.phase(phase_arr)
            time = time + chunklen*tres/86400.
        self.ts_pulses = np.concatenate(split_ts_pulses)

        print "Time series complete."

    def plot(self, withnoise=True, rounded=False):
        """
        Show a plot of the time series.

        withnoise: include the noise in the plot
        rounded: round values to nearest integers as in a presto .dat file
        """
        axis = np.linspace(0, self.nbins*self.tres, self.nbins, endpoint=False)
        if withnoise and rounded:
            plot(axis, np.round(self.ts_pulses+self.ts_noise))
        elif withnoise and not rounded:
            plot(axis, self.ts_pulses+self.ts_noise)
        elif not withnoise and rounded:
            plot(axis, np.round(self.ts_pulses))
        elif not withnoise and not rounded:
            plot(axis, self.ts_pulses)
        else: sys.exit('What did you just do?')
        xlabel('Time (seconds)')
        xlim(0, len(self.ts_pulses)*self.tres)
        savefig('ts_plot.eps')

    def savedat(self, fname, withnoise=True):
        """
        Save a presto-readable .dat file (with a corresponding .inf file)
        containing the time series.

        fname: filename without suffix, to be used for .inf and .dat files
        """
        inf = infodata()
        inf.name = fname
        inf.telescope = 'NA'
        inf.instrument = 'NA'
        inf.object = self.profile.pars['PSR']
        inf.observer = 'NA'
        inf.mjd_i = self.start.day_int
        inf.mjd_f = self.start.day_frac
        inf.bary = int(self.bary)
        inf.N = self.nbins
        inf.dt = self.tres
        inf.band = 'Radio'
        inf.filt = 'NA'
        inf.analyzer = 'NA'
        inf.notes = 'Completely fake data set.'

        print "Writing %s.dat and %s.inf..." % (fname, fname)

        # round since .dat files use integers
        if withnoise:
            write_dat(np.round(self.ts_pulses+self.ts_noise), fname)
        else:
            write_dat(np.round(self.ts_pulses), fname)
        writeinf(inf)

        print "Done."

    def set_noise(self, noise):
        self.ts_noise = np.random.normal(loc=self.zero, scale=noise,\
                                         size=self.nbins)

    def ts(self, rounded=False):
        if rounded: return np.round(self.ts_pulses + self.ts_noise)
        else: return self.ts_pulses + self.ts_noise

    def __add__(self, otherTS):
        """
        For now, this will just return an array without noise that contains
        the pulses of both TimeSeries objects.  They must have the same length
        and tres of course.
        """
        if (self.tres != otherTS.tres) or (self.length != otherTS.length):
            sys.exit('When adding TimeSeries objects, they must have the same'+\
                     ' length and time resolution.')
        else:
            return self.ts_pulses + otherTS.ts_pulses

def gaussian(phasebins=1024.0, sigma=0.01, mean=0.5, amp=1.0, loval=0.0,\
             hival=1.0):
    """
    Generates a Gaussian function on axis between loval and hival.  If
    phasebins is a non-integer, the last bin will be completed, but hival
    will be taken as representing the fractional bin value.

    phasebins: number of bins used (see above for details)
    sigma: standard deviation of the Gaussian
    mean: position of the Gaussian
    amp: height of the Gaussian
    loval: starting value of the x-axis
    hival: end value of the x-axis
    """
    nbins = np.ceil(phasebins)
    hival_bins = (nbins/phasebins)*(hival-loval)+loval
    x = np.linspace(loval, hival_bins, nbins, endpoint=False)
    return amp*np.exp(-np.square(x-mean)/(2.*sigma*sigma))

#                            _/              _/_/_/           
#       _/_/_/      _/_/    _/  _/    _/  _/          _/_/    
#      _/    _/  _/    _/  _/  _/    _/  _/        _/    _/   
#     _/    _/  _/    _/  _/  _/    _/  _/        _/    _/    
#    _/_/_/      _/_/    _/    _/_/_/    _/_/_/    _/_/       
#   _/                            _/                          
#  _/                        _/_/                             

class polyCo:
    """
    profile: psrProfile object for which to generate polycos
    start: starting time as an MJD object
    tobs: durating of polyco-usage in seconds, but will be changed to be an
        integer number of minutes (less than the input value) if it is not a
        multiple of 60
    obs: observatory code (-1 is barycentred)
    freq: radio frequency for which to generate polycos
    ncoeff: number of coefficients to generate
    delete_outfile: if False, 'temp_polyco.dat' will remain in directory
    """
    def __init__(self, profile, start, tobs, obs=-1, freq=1420.0, ncoeff=12,\
        delete_outfile=True):
        os.system(('tempo -f %s -Z START=%s -Z SPAN=%f -Z TOBS=%fS -Z OBS=%s'+\
            ' -Z FREQ=%f -Z NCOEFF=%d -Z OUT=temp_polyco.dat') %\
            (profile.parfile, start.show(), tobs, tobs, obs, freq, ncoeff))

        polyfile = np.loadtxt('temp_polyco.dat', dtype=str, delimiter='$$$')

        self.start = MJD(start)
        self.tmid = self.start + np.floor(tobs/60.)/2880.
        self.rphase = float(polyfile[1].split()[0])
        self.polycos = []
        for line in polyfile[2:]:
            self.polycos += line.replace('D', 'E').split()
        self.polycos = np.array(self.polycos).astype(float)

        self.profile = profile
        self.tobs = tobs
        self.obs = obs
        self.freq = freq
        self.ncoeff = ncoeff

        self.end = self.start + np.floor(tobs/60.)/1440.

        if delete_outfile: os.system('rm temp_polyco.dat')

    def calc_phases(self, tres, nbins):
        """
        tres in seconds
        """
        dt_A = np.linspace(0., tres*nbins, nbins, endpoint=False)/86400.
        if self.start + dt_A[-1] > self.end:
            print "WARNING: Phases are being generated outside the range"+\
                " of validity of the set of polycos being used."
        dt = 1440.*(dt_A - float(self.tmid - self.start))
        return self.rphase + dt*60.*self.profile.f0 + polyval(dt, self.polycos)

    def calc_freq(self, time):
        """
        time: MJD object
        """
        dt = float(MJD(time)-self.tmid)*1440.
        multy = np.arange(1, self.ncoeff, 1)
        return self.profile.f0 + (1./60.)*polyval(dt, multy*self.polycos[1:])

def simple_fold(ts, tres_ts, p0, p1, phasebins, timebins=1):
    """
    ts: a time series
    tres_ts: the time resolution of the provided time series
    p0: period to fold
    p1: period derivative
    phasebins: number of bins to use across given period
    timebins: number of consecutive 1D profiles to produce
    """
    length_ts = len(ts)*tres_ts
    t_axis_ts = np.linspace(0, length_ts, len(ts), endpoint=False)
    t_axis_fold = []
    time = 0.
    while time < length_ts:
        per = p0 + p1*time
        pulse = np.linspace(time, time+per, phasebins, endpoint=False)
        t_axis_fold += list(pulse)
        time += per
    t_axis_fold = np.array(t_axis_fold)
    nbins_fold = len(t_axis_fold)
    if nbins_fold%(phasebins*timebins):
        nbins_fold += phasebins*timebins - nbins_fold%(phasebins*timebins)
    downsampled = np.interp(t_axis_fold, t_axis_ts, ts, right=0)
    wrapped = downsampled.reshape((-1, phasebins))
    summed = []
    for ii in range(timebins):
        n_to_sum = len(wrapped)/timebins
        summed.append(np.sum(wrapped[ii*n_to_sum:(ii+1)*n_to_sum], axis=0))
    summed = np.array(summed)
    if timebins > 1:
        figure(figsize=(5, 6))
        imshow(np.concatenate((summed, summed), axis=1),\
                   aspect=2.8*phasebins/timebins, cmap=cm.Greys,\
                   interpolation='nearest', origin='lower')
        xlabel('Phase bins')
        ylabel('Time bins')
        return summed
    else:
        plot(summed[0])
        xlabel('Phase bins')
        ylabel('Power (arbitrary units)')
        return summed[0]

def multi_psr_ts(psr_list, amp_list, start, tres, noise, length, fname,\
                 fpath='.', obs=-1):
    """
    fname: no extension
    fpath: directory to place dat and inf files into
    psr_list: iterable set of psrProfile objects
    amp_list: corresponding list of pulse heights as fraction of noise sigma
      (if no noise is present, values are simply used as heights)
    start: start time as MJD object
    noise: sigma of white noise in time series
    """
    nbins = int(np.ceil(length/tres))
    if nbins % 2: nbins -= 1

    full_ts = np.zeros(nbins)

    amp_list = np.array(amp_list)
    if noise: amp_list *= noise

    for ii in range(len(psr_list)):
        ts = TimeSeries(psr_list[ii], amp_list[ii], start, tres, 0, length,\
            obs)
        full_ts += ts.ts_pulses
        
    if noise: full_ts += np.random.normal(loc=0, scale=noise, size=nbins)

    print "Writing %s.dat and %s.inf..." % (fname, fname)

    write_dat(np.round(full_ts), '%s/%s' % (fpath, fname))

    inf = infodata()
    inf.name = fname
    inf.telescope = 'NA'
    inf.instrument = 'NA'
    inf.object = 'NA' 
    inf.observer = 'NA'
    inf.mjd_i = start.day_int
    inf.mjd_f = start.day_frac
    inf.bary = 1
    inf.N = nbins
    inf.dt = tres
    inf.band = 'Radio'
    inf.filt = 'NA'
    inf.analyzer = 'NA'
    inf.notes = 'Completely fake data set.'
    writeinf(inf)

    print "Done."

def multi_psr_ts_add(psr_list, amp_list, in_fname, out_fname, in_fpath='.',\
    out_fpath='.'):
    """
    edit of multi_psr_ts to allow adding to existing datfile...
    in_fname and out_fname: no extension
    amp_list: amps as fraction of time series 1-sigma noise level
    """
    in_inffile = read_inffile('%s/%s' % (in_fpath, in_fname))

    nbins = in_inffile.N
    tres = in_inffile.dt
    length = nbins*tres
    start = MJD(in_inffile.mjd_i, in_inffile.mjd_f)

    full_ts = load_dat('%s/%s' % (in_fpath, in_fname))

    amp_list = np.array(amp_list)
    noise = np.std(full_ts)
    amp_list *= noise

    for ii in range(len(psr_list)):
        ts = TimeSeries(psr_list[ii], amp_list[ii], start, tres, 0, length)
        full_ts += ts.ts_pulses
    
    print "Writing %s.dat and %s.inf..." % (out_fname, out_fname)

    write_dat(np.round(full_ts), '%s/%s' % (out_fpath, out_fname))

    inf = in_inffile
    inf.name = out_fname
    inf.observer = 'Nobody (edited data)'
    inf.analyzer = 'Nobody (edited data)'
    inf.notes = 'Edited file %s.dat' % in_fname
    writeinf(inf)

    print "Done."

def prof_samples(w=3, h=3, maxpeaks=5):
    """
    Generate w*h random profiles and plot them, to see how the profiles are
    looking.

    w = number of columns
    h = number of rows
    maxpeaks = maximum number of Gaussians used to construct profiles
    """
    fw = w*2.4
    fh = h*1.8
    if fw > 12.0: fw = 12.0
    if fh > 9.0: fh = 9.0
    figure(figsize=(fw, fh))
    subplots_adjust(bottom=.01, left=.01, right=.99, top=.99,\
        wspace=.05, hspace=.05)
    for ii in range(w*h):
        prof = psrProfile(maxpeaks=maxpeaks)
        profile = prof.bin_prof()
        subplot(h,w,ii+1)
        plot(profile)
        xlim(0, len(profile))
        ylim(-0.1, 1.1)
        xticks(())
        yticks(())
    savefig('prof_samples.eps', bbox_inches='tight')

def fft_ts(ts, hifreq=100.0):
    """
    Read in a TimeSeries object and show the spectrum.

    ts: a TimeSeries object
    hifreq: highest frequency to be plotted (Hz)
    """
    spectrum = np.abs(np.fft.rfft(ts.ts_pulses + ts.ts_noise))
    freqaxis = np.linspace(0, 0.5/ts.tres, len(spectrum))
    cutoff = np.ceil(len(freqaxis)*(2.*ts.tres*hifreq))
    plot(freqaxis[:cutoff], spectrum[:cutoff])
    xlim(0, hifreq)
    xlabel('Frequency (Hz)')

def load_dat(datfile):
    """
    Load a presto .dat time series into python as a numpy array.

    datfile: name of file to be read, excluding '.dat' at the end
    """
    infile = open(datfile+'.dat', 'r')
    rawdat = infile.read()
    infile.close()
    ts_len = str(len(rawdat)/4)
    ts = struct.unpack(ts_len+'f', rawdat)
    return np.array(ts)

def write_dat(ts, datfile):
    """
    Write a time series array to a presto .dat file.

    ts: time series (an array, not a TimeSeries object)
    datfile: name of file to be written, excluding '.dat' at the end
    """
    outstr = ''
    for ii in range(len(ts)):
        outstr += struct.pack('f', ts[ii])
    outfile = open(datfile+'.dat', 'w')
    outfile.write(outstr)
    outfile.close()

# with both masses and the 5 keplerian parameters (Pb, x, om, T0, e) we should
# have enough information to use DDGR...
def create_parfile(name, p0, p1, posepoch, pepoch, T0, Pb,\
                   om=None, ecc=None, incl=None, m_p=1.4, m_c=1.4,\
                   comments=None, **kwargs):
    """
    name: pulsar name
    p0: spin period (s)
    p1: spin period derivative
    posepoch, pepoch, T0: MJD objects for the various parameter epochs
    Pb: binary period in days
    om: argument of periastron in degrees
    ecc: orbital eccentricity
    incl: inclination angle in degrees between 0 and 90
    m_p, m_c: masses (solar units) of pulsar and companion
    comments: string to be appended to parfile as comment

    kwargs
    ------
    tzrmjd: default same as pepoch, MJD object used as TOA for reference phase
    tzrfrq: default 1420.0, reference observing frequency
    tzrsite: default -1 (barycentre), reference site
    """
    pars = ['PSR', 'RAJ', 'DECJ', 'POSEPOCH', 'P0', 'P1', 'PEPOCH', 'DM',\
            'BINARY', 'A1', 'E', 'T0', 'PB', 'OM', 'MTOT', 'M2',\
            'OMDOT', 'GAMMA', 'PBDOT', 'SINI', 'TZRMJD', 'TZRFRQ', 'TZRSITE']

    posepoch = MJD(posepoch)
    pepoch = MJD(pepoch)
    T0 = MJD(T0)

    tzrmjd = kwargs.get('tzrmjd', pepoch)
    tzrfrq = kwargs.get('tzrfrq', 1420.0)
    tzrsite = kwargs.get('tzrsite', -1)

    # Get a random inclination if none is given
    if incl is None:
        cosi = np.random.uniform()
        sini = np.sqrt(1. - cosi*cosi)
    else:
        cosi = np.cos(incl*np.pi/180.)
        sini = np.sin(incl*np.pi/180.)

    # Get a_p from Pb + Kepler's laws
    Pbsq = (86400.*Pb)**2
    a_tot = pow(Pbsq*TSUN*(m_p+m_c)/(4.*np.pi*np.pi), 1./3.)
    a_p = a_tot*m_c/(m_p+m_c)

    # Get x = asini
    asini = a_p*sini

    # Get a random eccentricity if none is given
    if ecc is None: ecc = np.random.uniform()

    # Get a random longitude of periastron is none is given
    if om is None: om = np.random.uniform(high=360.)

    # PK parameters (except m_c which we already have as input)
    n = 2.*np.pi/(Pb*86400.)
    omdot = 3*pow(n,5./3.)*pow(((m_p+m_c)*TSUN),2./3.)/(1.-ecc*ecc)/OMFAC
    gamma = pow(n,-1./3.)*ecc*m_c*(2.*m_c+m_p)*pow((m_p+m_c),-4./3.)*\
        pow(TSUN,2./3.)
    pbdot = -192.*np.pi/5.*pow(n,5./3.)*m_p*m_c*pow((m_p+m_c),-1./3.)*\
        pow(TSUN,5./3.)*(1.+73./24.*ecc*ecc+37./96.*pow(ecc,4))*\
        pow((1.-ecc*ecc),-3.5)
    sini_out = asini*pow(n,2./3.)*pow((m_p+m_c),2./3.)*pow(TSUN,-1./3.)/m_c

    par_vals = [name, '00:00:00.0', '00:00:00.0', posepoch.show(), repr(p0),\
                repr(p1), pepoch.show(), '100.0', 'DD',\
                repr(asini), repr(ecc), T0.show(), repr(Pb), repr(om),\
                repr(m_p+m_c), repr(m_c), repr(omdot), repr(gamma),\
                repr(pbdot), repr(sini_out), repr(tzrmjd), repr(tzrfrq),\
                str(tzrsite)]

    parline = '%-9s %22s'
    out_lines = []
    for ii in range(len(pars)):
        out_lines.append(parline % (pars[ii], par_vals[ii]))
    if comments is not None:
        out_lines.append('# ' + str(comments))

    return out_lines
