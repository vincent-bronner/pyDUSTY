import numpy as np
import os
from astropy import units as u
from astropy import constants as const

Wm2_to_cgs = 1e3
Lsun = const.L_sun.cgs.value
clight = const.c.cgs.value
pc_to_cm = (1*u.pc).to('cm').value

class dusty_reader():
    def __init__(self, filename, luminosity=None, distance=10):
        self.filename = filename
        self.model_name = os.path.basename(filename).replace('.out', '')
        if luminosity is not None:
            self.luminosity = luminosity # in Lsun
        self.distance = distance  # in pc
        self._read_out()
        self._read_spp()
        self._read_stb()
        self._read_rtb()

    def _read_out(self):
        # read the raw file to find the number of models
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # get the line numbers where the new models start
        i_start = -1
        i_end = -1
        bad_lines = 0
        for i, line in enumerate(lines):
            i = i+1 # count from 1
            # check if the line contains "RESULTS:"
            if 'RESULTS:' in line:
                i_start = i + 4  # 4 extra lines to skip 

            if ('** WARNING' in line) or ('*The point-source assumption' in line):
                bad_lines += 1
            
            
            # check if the line contains "(1) Optical depth at 5.5E-01 microns"
            if '(1) Optical depth at 5.5E-01 microns' in line:
                i_end = i - 2 # 2 line to skip
        self.N_models = i_end - i_start - bad_lines
        print(f'Found {self.N_models} models in {self.model_name}')

        out_data = np.loadtxt(self.filename, skiprows=i_start, max_rows=self.N_models, unpack=True, comments=['#', '*'])

        # unpack the data
        self.tau = out_data[1]
        self.F1 = out_data[2] * Wm2_to_cgs
        self.r1 = out_data[3]
        self.r1_div_rstar = out_data[4]
        self.theta1 = out_data[5]
        self.Td_Y = out_data[6]

        # adjust r1 of luminosity is provided
        if hasattr(self, 'luminosity'):
            self.r1 = self.r1 * np.sqrt(self.luminosity / 1e4)

    def _read_spp(self):
        spp_filename = self.filename.replace('.out', '.spp')
        # check if the file exists
        if not os.path.exists(spp_filename):
            print('File not found: {}'.format(spp_filename))
            return

        # read the file if it exists
        spp_data = np.genfromtxt(spp_filename, unpack=True)

        # unpack the data
        self.Psi = spp_data[2]
        self.fV = spp_data[3]
        self.fK = spp_data[4]
        self.f12 = spp_data[5]
        self.C21 = spp_data[6]
        self.C31 = spp_data[7]
        self.C43 = spp_data[8]
        self.v8_13 = spp_data[9]
        self.b14_22 = spp_data[10]
        self.B9p8 = spp_data[11]
        self.B11p4 = spp_data[12]
        self.R9p8_18 = spp_data[13]

    
    def _read_stb(self):
        stb_filename = self.filename.replace('.out', '.stb')
        # check if the file exists
        if not os.path.exists(stb_filename):
            print('File not found: {}'.format(stb_filename))
            return

        # read the file if it exists
        self._stb_data = np.genfromtxt(stb_filename, unpack=True)

        # get the indices where the new spectra start by checking where the
        # wavelength decreases and the new spectrum starts
        self._stb_start = np.where(np.diff(self._stb_data[0], prepend=0) < 0)[0]
        self._stb_bounds = np.array([0] + self._stb_start.tolist() + [len(self._stb_data[0])])

        # get the names for the columns in the spectra
        self.stb_names = [
            'lambda', 'fTot', 'xAtt', 'xDs', 'xDe', 'fInp', 'tauT', 'albedo'
        ]

        # if the luminosity is provided, compute the correct SED and F_nu
        if hasattr(self, 'luminosity'):
            # compute the SED (lambda * F_lambda = nu * F_nu) in erg/s/cm^2
            SED = self._stb_data[1] * self.luminosity * Lsun / (4*np.pi*(self.distance*pc_to_cm)**2)

            # convert the flux to F_nu
            lamb = self._stb_data[0]*1e-4 # convert to cm
            F_nu = SED * lamb / clight

            # convert the flux to F_lam
            lam_AA = lamb * 1e8 # convert to AA
            F_lam = SED / lam_AA

            # store the new SED and F_nu in _stb_data
            self._stb_data = np.vstack((self._stb_data, SED, F_nu, F_lam))
            self.stb_names += ['SED', 'F_nu', 'F_lam']

            
    
    def spectrum(self, qty, model=-1, tau=None):
        # return the spectrum for a given quantity for the model number or tau

        # check if the stb file has been read
        if not hasattr(self, '_stb_start'):
            print('No spectra found for the model')
            return
        
        # check if the quantity is in the list of quantities
        assert qty in self.stb_names, f'Quantity not found in the spectra. Available quantities are: {self.stb_names}'

        # get the index of the quantity
        qty_idx = self.stb_names.index(qty)

        # get the index of the model if tau is specified
        if (tau is not None) and (model == -1):
            model = np.argmin(np.abs(self.tau - tau))
        elif (tau is not None) and (model != -1):
            print('Cannot specify both model and tau')
            return
        elif model == -1:
            model = self.N_models - 1
        
        return self._stb_data[qty_idx][self._stb_bounds[model]:self._stb_bounds[model+1]]


    def _read_rtb(self):
        rtb_filename = self.filename.replace('.out', '.rtb')
        # check if the file exists
        if not os.path.exists(rtb_filename):
            print('File not found: {}'.format(rtb_filename))
            return

        # read the file if it exists
        self._rtb_data = np.genfromtxt(rtb_filename, unpack=True)

        # get the indices where the new profiles start by checking where the
        # radius decreases and the new profile starts
        self._rtb_start = np.where(np.diff(self._rtb_data[0], prepend=0) < 0)[0]
        self._rtb_bounds = np.array([0] + self._rtb_start.tolist() + [len(self._rtb_data[0])])

        # get the names for the columns in the profiles
        self.rtb_names = [
            'y', 'eta', 't', 'tauF', 'epsilon', 'Td', 'rg'
        ]

        # set rho1 in units of the opacity kappa_mu0
        self.rho1 = self.tau / self.r1

    def profile(self, qty, model=-1, tau=None):
        # return the profile for a given quantity for the model number or tau

        # check if the rtb file has been read
        if not hasattr(self, '_rtb_start'):
            print('No profiles found for the model')
            return
        
        # check if the quantity is in the list of quantities
        assert qty in self.rtb_names, f'Quantity not found in the profiles. Available quantities are: {self.rtb_names}'

        # get the index of the quantity
        qty_idx = self.rtb_names.index(qty)

        # get the index of the model if tau is specified
        if (tau is not None) and (model == -1):
            model = np.argmin(np.abs(self.tau - tau))
        elif (tau is not None) and (model != -1):
            print('Cannot specify both model and tau')
            return
        elif model == -1:
            model = self.N_models - 1
        
        return self._rtb_data[qty_idx][self._rtb_bounds[model]:self._rtb_bounds[model+1]]
    
