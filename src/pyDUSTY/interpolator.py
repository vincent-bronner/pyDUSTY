import numpy as np
import re
from glob import glob
from scipy.interpolate import LinearNDInterpolator as lndi
from scipy.interpolate import interp1d


# define a summary reader
class summary_reader():
    def __init__(self, T, logg, model_dir, M=5.0, t=5):
        self.T = T
        self.logg = logg
        self.path = f'{model_dir}/output/T{T:.0f}_logg{logg:+.1f}_M{M:.1f}_t0{t:d}.summary'
        self.data = np.loadtxt(self.path, skiprows=3)
        self.header = ['tau', 'r1', 'eta1', '36', '45', 'J', 'H', 'Ks', 'U', 'B', 'V', 'R', 'I']

    def get(self, key):
        i = self.header.index(key)
        return self.data[:, i]
    
    def get_t(self, key, tau):
        return np.interp(tau, self.get('tau'), self.get(key))

    def get_data(self):
        data = self.data.copy()
        data_flat = data.flatten()
        return data_flat
    



# light curve interpolator
class LC_interp():
    def __init__(self, Teff, logg, logL, model_dir, M=5.0, t=5):
        self.Teff = Teff # array containing the effective temperature in K
        self.logg = logg # array containing the log g in cm
        self.L = 10**logL # array containing the L in Lsun
        self.model_dir = model_dir
        self.M = M
        self.t = t
        self.data_header = ['tau', 'r1', 'eta1', '36', '45', 'J', 'H', 'Ks', 'U', 'B', 'V', 'R', 'I']
        self.base_dir = f'{self.model_dir}/output/'
        self.paths = sorted(glob(f'{self.base_dir}*.summary'))
        self.get_T_logg_vals()
        self.build_interpolator()
        self.do_interpolation()

    def get_evo(self, tau0, qty, rho_base_inp=None):
        tau_evo = self.get_tau_evo(tau0, rho_base_inp)
        qty_evo = self.get_qty_evo(qty, tau_evo)
        return qty_evo

    def get_T_logg_vals(self):
        T_logg = []
        for p in self.paths:
            T, logg = re.search('T(\d+)_logg([+-]\d+\.\d+)', p).groups()
            T_logg.append([float(T), float(logg)])
        self.T_logg = np.array(T_logg)
    
    def build_interpolator(self):
        # get interpolation data
        interp_data = []
        for T, logg in self.T_logg:
            r = summary_reader(T, logg, model=self.model, M=self.M, t=self.t)
            interp_data.append(r.get_data())
        interp_data = np.array(interp_data)

        # get interpolation function
        self.interp = lndi(self.T_logg, interp_data, fill_value=np.nan)

    def do_interpolation(self):
        # get the interpolated values
        interp_vals = self.interp(self.Teff, self.logg)
        interp_vals = interp_vals.reshape(len(self.Teff), -1, 13)

        # scale everything to the correct luminosity
        # r1 = r1 * sqrt(L / 1e4)  (DUSTY computes r1 for L = 1e4 Lsun)
        i_r1 = self.data_header.index('r1')
        interp_vals[:, :, i_r1] *= np.sqrt(self.L[:, None] / 1e4)

        # mags = mags - 2.5 * log10(L)  (magnitudes are for L = 1 Lsun)
        for qty in ['36', '45', 'J', 'H', 'Ks', 'U', 'B', 'V', 'R', 'I']:
            i = self.data_header.index(qty)
            interp_vals[:, :, i] -= 2.5 * np.log10(self.L[:, None])

        self.interp_vals = interp_vals

        # get array of densities at r=1e16 cm
        r_rho = 1e16
        i_tau = self.data_header.index('tau')
        i_r1 = self.data_header.index('r1')
        r1 = interp_vals[:, :, i_r1]
        rho1 = interp_vals[:, :, i_tau] / r1
        self.rho_r = rho1 * (r1 / r_rho)**2


    def get_tau_evo(self, tau0, rho_base_inp=None):
        # compute the evolution of tau for a given tau0
        i_tau = self.data_header.index('tau')

        if rho_base_inp is None:
            # get the rho_r value for tau0
            interp_rho0 = interp1d(self.interp_vals[0, :, i_tau], self.rho_r[0], fill_value=np.nan)
            rho_base = interp_rho0(tau0)
            assert not(np.isnan(rho_base)), f'rho_base is nan for tau0 = {tau0}'
        else:
            rho_base = rho_base_inp
        self.rho_base = rho_base

        # get the tau evolution
        tau_evo = np.zeros_like(self.Teff)
        for i, rho_r in enumerate(self.rho_r):
            interp_rho = interp1d(rho_r, self.interp_vals[i, :, i_tau], fill_value=np.nan)
            tau_evo[i] = interp_rho(rho_base)
        return tau_evo

    def get_qty_evo(self, qty, tau_evo):
        i_qty = self.data_header.index(qty)
        i_tau = self.data_header.index('tau')
        qty_evo = np.zeros_like(self.Teff)
        for i, tau in enumerate(tau_evo):
            interp_qty = interp1d(self.interp_vals[i, :, i_tau], self.interp_vals[i, :, i_qty], fill_value=np.nan)
            qty_evo[i] = interp_qty(tau)
        return qty_evo