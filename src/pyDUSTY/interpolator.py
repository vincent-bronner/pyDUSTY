import numpy as np
import re
from glob import glob
from scipy.interpolate import LinearNDInterpolator as lndi
from scipy.interpolate import interp1d


# define a summary reader
class summary_reader():
    def __init__(self, T, logg, model_dir, M=5.0, t=5):
        """Read the summary file for a given model of T and logg in the model_dir

        Parameters
        ----------
        T : float
            Effective temperature of the model
        logg : float
            log g of the model
        model_dir : str
            Directory containing the model files
        M : float, optional
            mass, by default 5.0
        t : int, optional
            microturbulence parameter, by default 5
        """

        self.T = T
        self.logg = logg
        self.path = f'{model_dir}/output/T{T:.0f}_logg{logg:+.1f}_M{M:.1f}_t0{t:d}.summary'
        self.data = np.loadtxt(self.path, skiprows=3)
        self.header = ['tau', 'r1', 'eta1', '36', '45', 'J', 'H', 'Ks', 'U', 'B', 'V', 'R', 'I']


    def get(self, key):
        """
        Retrieve data corresponding to the specified key.
        Args:
            key (str): The key to look up in the header.
        Returns:
            numpy.ndarray: The data array corresponding to the specified key.
        """

        i = self.header.index(key)
        return self.data[:, i]
    

    def get_t(self, key, tau):
        """
        Interpolates the value of the specified key at a given tau.
        Parameters:
        key (str): The key for which the value needs to be interpolated.
        tau (float): The tau value at which interpolation is to be performed.
        Returns:
        float: The interpolated value corresponding to the given key and tau.
        """

        return np.interp(tau, self.get('tau'), self.get(key))


    def get_data(self):
        """
        Retrieves a flattened copy of the data.
        Returns:
            numpy.ndarray: A flattened copy of the data.
        """
        
        data = self.data.copy()
        data_flat = data.flatten()
        return data_flat
    



# light curve interpolator
class DM_interpolator():
    def __init__(self, Teff, logg, logL, model_dir, M=5.0, t=5):
        """Interpolates the DUSTY model data to get the light curve for a given set of Teff, logg, and logL values.

        Parameters
        ----------
        Teff : numpy.ndarray
            Array containing the effective temperature in K
        logg : numpy.ndarray
            Array containing the log g in cm s^-2
        logL : numpy.ndarray
            Array containing the log L in Lsun
        model_dir : str
            Directory containing the model files
        M : float, optional
            mass, by default 5.0
        t : int, optional
            microturbulence parameter, by default 5
        """
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
        """
        Computes the evolution of a given quantity over optical depth.
        Parameters:
        -----------
        tau0 : float
            Initial optical depth.
        qty : array-like
            The quantity to evolve.
        rho_base_inp : float, optional
            Base density input, by default None.
        Returns:
        --------
        array-like
            The evolved quantity over optical depth.
        """
        
        tau_evo = self.get_tau_evo(tau0, rho_base_inp)
        qty_evo = self.get_qty_evo(qty, tau_evo)
        return qty_evo



    def get_T_logg_vals(self):
        """
        Extracts temperature (T) and logarithm of surface gravity (logg) values from file paths.
        This method searches through the list of file paths stored in `self.paths`, extracts the 
        temperature and logg values using a regular expression, and stores them as a NumPy array 
        in `self.T_logg`.
        The expected format in the file paths is 'T<temperature>_logg<logg>', where <temperature> 
        is an integer and <logg> is a floating-point number.
        Raises:
            AttributeError: If `self.paths` is not defined.
            ValueError: If the regular expression does not match the expected format in any of the paths.
        """
        
        T_logg = []
        for p in self.paths:
            T, logg = re.search('T(\d+)_logg([+-]\d+\.\d+)', p).groups()
            T_logg.append([float(T), float(logg)])
        self.T_logg = np.array(T_logg)
    


    def build_interpolator(self):
        """
        Builds an interpolator for the given temperature and log(g) values.
        This method performs the following steps:
        1. Retrieves interpolation data for each pair of temperature (T) and log(g) values.
        2. Reads the summary data using the `summary_reader` function and appends it to the interpolation data list.
        3. Converts the interpolation data list to a NumPy array.
        4. Creates an interpolation function using the `lndi` function with the temperature and log(g) values and the interpolation data.
        The resulting interpolator is stored in the `self.interp` attribute.
        Attributes:
        -----------
        self.T_logg : list of tuples
            List of tuples containing temperature (T) and log(g) values.
        self.model_dir : str
            Directory containing the model data.
        self.M : float
            Mass parameter used in the summary reader.
        self.t : float
            Time parameter used in the summary reader.
        self.interp : function
            Interpolation function created using the temperature and log(g) values and the interpolation data.
        """

        # get interpolation data
        interp_data = []
        for T, logg in self.T_logg:
            r = summary_reader(T, logg, model_dir=self.model_dir, M=self.M, t=self.t)
            interp_data.append(r.get_data())
        interp_data = np.array(interp_data)

        # get interpolation function
        self.interp = lndi(self.T_logg, interp_data, fill_value=np.nan)



    def do_interpolation(self):
        """
        Perform interpolation and scaling of stellar parameters.
        This method interpolates the effective temperature (Teff) and surface gravity (logg)
        to obtain interpolated values. It then scales these values to the correct luminosity (L)
        and adjusts magnitudes accordingly. Additionally, it computes the density at a specific
        radius (r=1e16 cm).
        Steps:
        1. Interpolate Teff and logg to get interpolated values.
        2. Reshape the interpolated values.
        3. Scale the radius (r1) to the correct luminosity.
        4. Adjust magnitudes for the correct luminosity.
        5. Compute the density at r=1e16 cm.
        Attributes:
            interp_vals (numpy.ndarray): The interpolated and scaled values.
            rho_r (numpy.ndarray): The density at r=1e16 cm.
        Raises:
            ValueError: If 'r1' or 'tau' is not found in data_header.
        """

        # get the interpolated values
        interp_vals = self.interp(self.Teff, self.logg)
        interp_vals = interp_vals.reshape(len(self.Teff), -1, 13)

        # scale everything to the correct luminosity
        # r1 = r1 * sqrt(L / 1e0)  (DUSTY computes r1 for L = 1e4 Lsun, but summary files are for L = 1 Lsun)
        i_r1 = self.data_header.index('r1')
        interp_vals[:, :, i_r1] *= np.sqrt(self.L[:, None] / 1e0)

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
        """
        Compute the evolution of tau for a given initial tau value (tau0).
        Parameters:
        -----------
        tau0 : float
            The initial tau value for which the evolution is to be computed.
        rho_base_inp : float, optional
            The base density value. If not provided, it will be interpolated based on tau0.
        Returns:
        --------
        tau_evo : numpy.ndarray
            An array containing the evolution of tau values corresponding to the effective temperatures (Teff).
        Raises:
        -------
        AssertionError
            If the interpolated base density (rho_base) is NaN for the given tau0.
        """
        
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
        """
        Interpolates the evolution of a given quantity over optical depth values.
        Parameters:
        -----------
        qty : str
            The name of the quantity to interpolate.
        tau_evo : array-like
            An array of optical depth values at which to interpolate the quantity.
        Returns:
        --------
        qty_evo : numpy.ndarray
            An array containing the interpolated values of the specified quantity at the given optical depth values.
        """

        i_qty = self.data_header.index(qty)
        i_tau = self.data_header.index('tau')
        qty_evo = np.zeros_like(self.Teff)
        for i, tau in enumerate(tau_evo):
            interp_qty = interp1d(self.interp_vals[i, :, i_tau], self.interp_vals[i, :, i_qty], fill_value=np.nan)
            qty_evo[i] = interp_qty(tau)
        return qty_evo
