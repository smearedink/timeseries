import numpy as np
import sys, os, time, locale
from PySide import QtCore, QtGui
from psr_profile import *
from tseries_gui import Ui_TimeSeries

class TimeSeriesEditor(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = Ui_TimeSeries()
        self.ui.setupUi(self)

        self.setFixedSize(self.size())

        self.ui.inputStartTime.setText(str(current_mjd().day_int))

        self.status_norm_clr = QtGui.QColor(220, 220, 220)
        self.status_save_clr = QtGui.QColor(0, 255, 0)
        self.ui.statusreport.setText("Welcome to Time Series Editor! Enjoy faking pulsar data.")
        self.ui.statusreport.setStyleSheet("QFrame { background-color: %s }" % self.status_norm_clr.name())

        try: locale.setlocale(locale.LC_ALL, 'en_US')
        except: pass

        # Link buttons and things
        QtCore.QObject.connect(self.ui.AddPSRs,\
            QtCore.SIGNAL('clicked()'), self.add_pulsars)
        QtCore.QObject.connect(self.ui.RemovePSR,\
            QtCore.SIGNAL('clicked()'), self.remove_pulsar)
        QtCore.QObject.connect(self.ui.ClearPSRs,\
            QtCore.SIGNAL('clicked()'), self.remove_all_pulsars)
        QtCore.QObject.connect(self.ui.PSRlist,\
            QtCore.SIGNAL('itemSelectionChanged()'), self.show_pars)
        QtCore.QObject.connect(self.ui.PSRlist,\
            QtCore.SIGNAL('itemSelectionChanged()'), self.disable_savechanges)
        QtCore.QObject.connect(self.ui.SaveParChanges,\
            QtCore.SIGNAL('clicked()'), self.update_pars)
        QtCore.QObject.connect(self.ui.SaveParChanges,\
            QtCore.SIGNAL('clicked()'), self.disable_savechanges)
        self.parfields = [self.ui.inputPSR, self.ui.inputP0, self.ui.inputP1, self.ui.inputPOSEPOCH, self.ui.inputPEPOCH, self.ui.inputT0, self.ui.inputPB, self.ui.inputOM, self.ui.inputE, self.ui.inputINC, self.ui.inputM1, self.ui.inputM2, self.ui.inputAMP]
        for parfield in self.parfields:
            QtCore.QObject.connect(parfield,\
                QtCore.SIGNAL('textEdited(const QString &)'),\
                self.enable_savechanges)
        QtCore.QObject.connect(self.ui.AddPSRs,\
            QtCore.SIGNAL('clicked()'), self.enable_clearPSRs_makeTS_addToTS)
        QtCore.QObject.connect(self.ui.RemovePSR,\
            QtCore.SIGNAL('clicked()'), self.disable_clearPSRs_makeTS_addToTS)
        QtCore.QObject.connect(self.ui.RemovePSR,\
            QtCore.SIGNAL('clicked()'), self.disable_removePSR)
        QtCore.QObject.connect(self.ui.ClearPSRs,\
            QtCore.SIGNAL('clicked()'), self.disable_removePSR)
        QtCore.QObject.connect(self.ui.ClearPSRs,\
            QtCore.SIGNAL('clicked()'), self.disable_clearPSRs_makeTS_addToTS)
        QtCore.QObject.connect(self.ui.PSRlist,\
            QtCore.SIGNAL('itemSelectionChanged()'), self.enable_removePSR)
        QtCore.QObject.connect(self.ui.MakeTS,\
            QtCore.SIGNAL('clicked()'), self.new_TS)
        QtCore.QObject.connect(self.ui.AddToTS,\
            QtCore.SIGNAL('clicked()'), self.add_to_TS)

        # Keep track of pulsars in a simple way
        # (psrnum isn't really the total number of pulsars, it just keeps
        #  the label numbers in some sensible order)
        self.psrnum = 0
        self.psr_pars_dict = {}

    def enable_savechanges(self):
        self.ui.SaveParChanges.setEnabled(True)
    def disable_savechanges(self):
        self.ui.SaveParChanges.setEnabled(False)
    def enable_removePSR(self):
        if self.ui.PSRlist.count() != 0:
            self.ui.RemovePSR.setEnabled(True)
    def disable_removePSR(self):
        if self.ui.PSRlist.count() == 0:
            self.ui.RemovePSR.setEnabled(False)
    def enable_clearPSRs_makeTS_addToTS(self):
        if self.ui.PSRlist.count() != 0:
            self.ui.ClearPSRs.setEnabled(True)
            self.ui.MakeTS.setEnabled(True)
            self.ui.AddToTS.setEnabled(True)
    def disable_clearPSRs_makeTS_addToTS(self):
        if self.ui.PSRlist.count() == 0:
            self.ui.ClearPSRs.setEnabled(False)
            self.ui.MakeTS.setEnabled(False)
            self.ui.AddToTS.setEnabled(False)

    def add_to_TS(self):
        filewindow = QtGui.QFileDialog()
        msgbox = QtGui.QMessageBox()

        loadpath = str(filewindow.getOpenFileName()[0])
        if not loadpath: return
        elif loadpath[-4:] != '.dat':
            msgbox.setText('Loaded file must be .dat type')
            msgbox.exec_()
            return
        load_basename = loadpath.split('/')[-1][:-4]
        pathsplit = loadpath.split('/')[:-1]
        load_basedir = ''
        for word in pathsplit: load_basedir += (word + '/')
        if not os.path.exists(load_basedir + load_basename + '.inf'):
            msgbox.setText('.dat file must have corresponding .inf file')
            msgbox.exec_()
            return

        # Ask if user would like to change reference epochs to be close to
        # that of loaded time series
        load_inf = read_inffile(load_basedir + load_basename)
        load_mjd = int(load_inf.mjd_i)
        ask_mjd_question = False
        for psr in self.psr_pars_dict.values():
            for day in [psr.posepoch, psr.pepoch, psr.t0]:
                if MJD(day).day_int != load_mjd:
                    ask_mjd_question = True
                    break
        if ask_mjd_question:
            mjd_question = 'Change all pulsar epochs to MJD '+str(load_mjd)+' to match loaded time series?'
            mjd_reply = QtGui.QMessageBox.question(self, 'Change MJDS?', mjd_question, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel, QtGui.QMessageBox.Yes)
            if mjd_reply == QtGui.QMessageBox.Yes:
                for psr in self.psr_pars_dict.values():
                    psr.posepoch = repr(load_mjd)
                    psr.pepoch = repr(load_mjd)
                    psr.t0 = repr(load_mjd - psr.t0_subtract)
                self.show_pars()
            elif mjd_reply == QtGui.QMessageBox.Cancel:
                return

        savepath = str(filewindow.getSaveFileName()[0])
        if not savepath: return
        if savepath[-4:] != '.dat': savepath += '.dat'
        save_basename = savepath.split('/')[-1][:-4]
        pathsplit = savepath.split('/')[:-1]
        save_basedir = ''
        for word in pathsplit: save_basedir += (word + '/')
        pardir = '%s_parfiles' % (save_basedir + save_basename)
        if not os.path.exists(pardir): os.makedirs(pardir)

        # Update status box to show we're saving now
        npsrs = len(self.psr_pars_dict)
        nbins = int(load_inf.N)
        try: totalbins_str = locale.format("%d", nbins*npsrs, grouping=True)
        except: totalbins_str = repr(nbins*npsrs)
        self.ui.statusreport.setText('Generating a total of %s time series bins. Please wait...' % totalbins_str)
        self.ui.statusreport.setStyleSheet("QFrame { background-color: %s }" % self.status_save_clr.name())
        QtGui.qApp.processEvents()

        # Generate parfiles and dat/inf files
        numpsrs = self.ui.PSRlist.count()
        profiles = []
        profile_amps = []
        for psr in self.psr_pars_dict.values():
            parfile = create_parfile(psr.psr, float(psr.p0), float(psr.p1),\
                MJD(psr.posepoch), MJD(psr.pepoch), MJD(psr.t0),\
                float(psr.pb), float(psr.om), float(psr.e), float(psr.inc),\
                float(psr.m1), float(psr.m2), 'Amplitude: '+psr.amp)
            parfile_loc = '%s/%s_%s.par' % (pardir, save_basename, psr.psr)
            np.savetxt(parfile_loc, parfile, fmt='%s')
            profiles.append(psrProfile(parfile_loc))
            profile_amps.append(float(psr.amp))
        start_time = MJD(str(self.ui.inputStartTime.text()))
        tres = float(self.ui.inputTres.text())
        noise = float(self.ui.inputNoise.text())
        length = float(self.ui.inputLength.text())

        multi_psr_ts_add(profiles, profile_amps, load_basename, save_basename, load_basedir, save_basedir)
        os.rename('%s.inf'%save_basename, '%s.inf'%(save_basedir+save_basename))

        for prof in profiles:
            prof.plot(tres=8.192e-7,\
                outfile=pardir+'/'+save_basename+'_'+prof.pars['PSR']+'.eps')

        # Update status box to show we're done saving
        self.ui.statusreport.setText('Files written to %s' % save_basedir)
        self.ui.statusreport.setStyleSheet("QFrame { background-color: %s }" % self.status_norm_clr.name())

    def new_TS(self):
        filewindow = QtGui.QFileDialog()
        savepath = str(filewindow.getSaveFileName()[0])
        if not savepath: return
        if savepath[-4:] != '.dat': savepath += '.dat'
        basename = savepath.split('/')[-1][:-4]
        pathsplit = savepath.split('/')[:-1]
        basedir = ''
        for word in pathsplit: basedir += (word + '/')
        pardir = '%s_parfiles' % (basedir + basename)
        if not os.path.exists(pardir): os.makedirs(pardir)

        # Update status box to show we're saving now
        npsrs = len(self.psr_pars_dict)
        tres = float(self.ui.inputTres.text())
        length = float(self.ui.inputLength.text())
        nbins = int(np.ceil(length/tres))
        if nbins % 2: nbins -= 1
        try: totalbins_str = locale.format("%d", nbins*npsrs, grouping=True)
        except: totalbins_str = repr(nbins*npsrs)
        self.ui.statusreport.setText('Generating a total of %s time series bins. Please wait...' % totalbins_str)
        self.ui.statusreport.setStyleSheet("QFrame { background-color: %s }" % self.status_save_clr.name())
        QtGui.qApp.processEvents()

        # Generate parfiles and dat/inf files
        numpsrs = self.ui.PSRlist.count()
        profiles = []
        profile_amps = []
        for psr in self.psr_pars_dict.values():
            parfile = create_parfile(psr.psr, float(psr.p0), float(psr.p1),\
                MJD(psr.posepoch), MJD(psr.pepoch), MJD(psr.t0),\
                float(psr.pb), float(psr.om), float(psr.e), float(psr.inc),\
                float(psr.m1), float(psr.m2), 'Amplitude: '+psr.amp)
            parfile_loc = '%s/%s_%s.par' % (pardir, basename, psr.psr)
            np.savetxt(parfile_loc, parfile, fmt='%s')
            profiles.append(psrProfile(parfile_loc))
            profile_amps.append(float(psr.amp))
        start_time = MJD(str(self.ui.inputStartTime.text()))
        noise = float(self.ui.inputNoise.text())

        multi_psr_ts(profiles, profile_amps, start_time, tres, noise, length, basename, basedir)
        os.rename('%s.inf'%basename, '%s.inf'%(basedir+basename))

        for prof in profiles:
            prof.plot(tres=8.192e-7,\
                outfile=pardir+'/'+basename+'_'+prof.pars['PSR']+'.eps')

        # Update status box to show we're done saving
        self.ui.statusreport.setText('Files written to %s' % basedir)
        self.ui.statusreport.setStyleSheet("QFrame { background-color: %s }" % self.status_norm_clr.name())

    def add_pulsars(self):
        num = self.ui.NumPSRs.value()
        psrnames = []
        for ii in range(num):
            psrname = 'Pulsar%0*d' % (2, self.psrnum+ii+1)
            self.psr_pars_dict[psrname] = ParInputs(psrname)
            self.ui.PSRlist.addItem(psrname)
        self.psrnum += num

    def remove_pulsar(self):
        for item in self.ui.PSRlist.selectedItems():
            del self.psr_pars_dict[str(item.text())]
            self.ui.PSRlist.takeItem(self.ui.PSRlist.row(item))

    def remove_all_pulsars(self):
        for row in list(reversed(range(self.ui.PSRlist.count()))):
            self.ui.PSRlist.takeItem(row)
        self.psrnum = 0
        self.psr_pars_dict = {}

    def show_pars(self):
        try:
            item = str(self.ui.PSRlist.selectedItems()[0].text())
            self.ui.inputPSR.setText(item)
            self.ui.inputP0.setText(self.psr_pars_dict[item].p0)
            self.ui.inputP1.setText(self.psr_pars_dict[item].p1)
            self.ui.inputPOSEPOCH.setText(self.psr_pars_dict[item].posepoch)
            self.ui.inputPEPOCH.setText(self.psr_pars_dict[item].pepoch)
            self.ui.inputT0.setText(self.psr_pars_dict[item].t0)
            self.ui.inputPB.setText(self.psr_pars_dict[item].pb)
            self.ui.inputOM.setText(self.psr_pars_dict[item].om)
            self.ui.inputE.setText(self.psr_pars_dict[item].e)
            self.ui.inputINC.setText(self.psr_pars_dict[item].inc)
            self.ui.inputM1.setText(self.psr_pars_dict[item].m1)
            self.ui.inputM2.setText(self.psr_pars_dict[item].m2)
            self.ui.inputAMP.setText(self.psr_pars_dict[item].amp)
        except:
            self.ui.inputPSR.setText('')
            self.ui.inputP0.setText('')
            self.ui.inputP1.setText('')
            self.ui.inputPOSEPOCH.setText('')
            self.ui.inputPEPOCH.setText('')
            self.ui.inputT0.setText('')
            self.ui.inputPB.setText('')
            self.ui.inputOM.setText('')
            self.ui.inputE.setText('')
            self.ui.inputINC.setText('')
            self.ui.inputM1.setText('')
            self.ui.inputM2.setText('')
            self.ui.inputAMP.setText('')

    def update_pars(self):
        try:
            item = str(self.ui.PSRlist.selectedItems()[0].text())
            self.psr_pars_dict[item].p0 = str(self.ui.inputP0.text())
            self.psr_pars_dict[item].p1 = str(self.ui.inputP1.text())
            self.psr_pars_dict[item].posepoch=str(self.ui.inputPOSEPOCH.text())
            self.psr_pars_dict[item].pepoch = str(self.ui.inputPEPOCH.text())
            self.psr_pars_dict[item].t0 = str(self.ui.inputT0.text())
            self.psr_pars_dict[item].pb = str(self.ui.inputPB.text())
            self.psr_pars_dict[item].om = str(self.ui.inputOM.text())
            self.psr_pars_dict[item].e = str(self.ui.inputE.text())
            self.psr_pars_dict[item].inc = str(self.ui.inputINC.text())
            self.psr_pars_dict[item].m1 = str(self.ui.inputM1.text())
            self.psr_pars_dict[item].m2 = str(self.ui.inputM2.text())
            self.psr_pars_dict[item].amp = str(self.ui.inputAMP.text())

            newname = str(self.ui.inputPSR.text())
            if newname != item:
                self.psr_pars_dict[item].psr = newname
                self.psr_pars_dict[newname] = self.psr_pars_dict[item]
                del self.psr_pars_dict[item]
                self.ui.PSRlist.selectedItems()[0].setText(newname)

        except:
            msgbox = QtGui.QMessageBox()
            msgbox.setText("You can't do that and I hope you're smart enough"\
                           +" to figure out why because I was too lazy to"\
                           +" tell you.")
            msgbox.exec_()

class ParInputs():
    def __init__(self, name=''):
        self.psr = name
        self.all_random()

    def all_random(self):
        p0 = np.random.uniform(low=0.001, high=0.02)
        # p1 range reflects doppler shift due to cluster motion
        p1 = np.random.uniform(low=-1.e-18, high=1.e-18)
        self.p0 = repr(p0)
        self.p1 = repr(p1)

        self.om = repr(np.random.uniform(high=360.))

        self.e = repr(np.random.uniform())

        inc = np.arccos(np.random.uniform())*180./np.pi
        self.inc = repr(inc)

        # Pb from 10 minutes to 1 day
        pb = np.random.uniform(low=6./864., high=1.0)
        self.pb = repr(pb)

        mjdstring = str(current_mjd().day_int)
        self.posepoch = mjdstring
        self.pepoch = mjdstring
        self.t0_subtract = np.random.uniform()*pb
        self.t0 = (MJD(mjdstring) - self.t0_subtract).show()

        self.m1 = '1.4'

        randy = np.random.uniform()
        if randy < 0.8:
            self.m2 = repr(np.random.uniform(0.05, 3.0))
        else:
            self.m2 = repr(np.random.uniform(3.0, 10.0))

        # pulse amplitude as a fraction of noise
        self.amp = repr(round(np.random.triangular(0.001, 0.001, 0.01), 7))

    def make_parfile(self):
        p0 = float(self.p0)
        p1 = float(self.p1)
        posepoch = MJD(self.posepoch)
        pepoch = MJD(self.pepoch)
        t0 = MJD(self.t0)
        pb = float(self.pb)
        om = float(self.om)
        e = float(self.e)
        inc = float(self.inc)
        m1 = float(self.m1)
        m2 = float(self.m2)
        comments = 'Amplitude: '+self.amp

        parfile = create_parfile(self.psr, p0, p1, posepoch, pepoch, t0, pb,\
                  om, e, inc, m1, m2, comments)
        np.savetxt('TSE_'+self.psr+'.par', parfile, fmt='%s')
        
        #print "Wrote file %s" % (self.psr+'.par')

def current_mjd():
    year, month, day, hour, minute, second = time.gmtime()[:6]
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + (12 * a) - 3
    p = day + (((153 * m) + 2) // 5) + (365 * y)
    q = (y // 4) - (y // 100) + (y // 400) - 32045
    jdn = p + q
    fracsec = current_time()
    fracsec -= np.floor(fracsec)
    jd = MJD(jdn, (hour-12.)/24. + minute/1440. + (second+fracsec)/86400.)
    return jd - 2400000.5


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = TimeSeriesEditor()
    myapp.show()
    sys.exit(app.exec_())
