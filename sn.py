# Likelihood for Pantheon+ SNe data, following CosmoSIS implementation

import numpy as np
import os
import pandas as pd
import scipy.linalg as la
import astropy.constants
import scipy.integrate as intgr
import json

# Local
from cobaya.log import LoggedError
from cobaya.likelihoods.base_classes import DataSetLikelihood

_twopi = 2 * np.pi

use_abs_mag = True
record_like = False  #For evaluating likelihoods
record_theory = False #Comparing class and manual
debug_post = False # For debugging posterior

if record_like:
    if use_abs_mag:
        post_record_file = '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/bidenko/LCDM/evaluate/cobaya_sn_like.txt'
    else:
        post_record_file = '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/bidenko/LCDM/evaluate/cobaya_sn_like_noMb.txt'
    file_like = open(post_record_file, 'w', buffering=1)
if record_theory:
    file_theory =  '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/bidenko/LCDM/evaluate/class_manual_da.txt'
    file_distances = open(file_theory, 'w', buffering=1)
    file_distances.write("manual class difference")
if debug_post == True:
    file_lumdists_name = "/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/bidenko/LCDM/evaluate/debug_post.txt"
    file_lumdists = open(file_lumdists_name, 'w', buffering=1)
    root = "/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/bidenko/LCDM/evaluate/outputs/"
#file = open("/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_bidenko/my_like_shoes/debug_cond.txt", 'w')

# define functions for distance calculations
def h_z(z,*args):
    H0,om,pwr = args
    ol  = 1 - om
    h = ( H0 * ( om * (1+z)**3 + ol ) ** 0.5 ) ** pwr
    return h 
def da(z, H0,om,pwr = -1):
    d = intgr.quad(h_z,0.,z,args=(H0,om,pwr))[0]*(astropy.constants.c.value/1000.)/(1+z)
    return d

class SN(DataSetLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "SN"

    install_options = {"github_repository": "CobayaSampler/sn_data",
                       "github_release": "v1.3"}

    def init_params(self, ini):

        ### intrinsicdisp = 0
        assert not ini.float('intrinsicdisp', 0) and not ini.float('intrinsicdisp0', 0)

        ### Pecz = 0, = 0.001 if not found in .dataset file
        self.pecz = ini.float('pecz', 0.001)

        data_file = "/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat"
        data=pd.read_csv(data_file,delimiter = ' ')
        self.log.debug('Reading %s' % data_file)
        self.origlen = len(data)
        
        #cov_file = '/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov'
        #self.cov = np.reshape(pn.read_table(cov_file,sep=' ').values,(1701,1701))
        
        # Load needed data, vectorized 
        self.mb = data['m_b_corr'].values
        self.z = data['zHD'].values
        self.zhel = data['zHEL'].values

        # Redshift selection cuts and identify SH0Es distance calibrators (if needed)
        if self.use_shoes:
            self.cut = ((data['zHD']>0.01) | (data['IS_CALIBRATOR']==1))
            print("Using SH0ES")
        else: 
            self.cut = (data['zHD']>0.01) 
        
        # Apply selection cuts
        self.zCMB = self.z[self.cut]
        self.mag = self.mb[self.cut]
        self.zHEL = self.zhel[self.cut]

        if self.use_shoes:
            self.is_calibrator = data['IS_CALIBRATOR'].values[self.cut]==1 # Boolean array
            self.cepheid_distance = data['CEPH_DIST'].values[self.cut]

        if not self.use_abs_mag: 
            # Use this for analytic marginalization
            self.mag_var = data['m_b_corr_err_DIAG'][self.cut]
        
        covmats = [
            'mag', 'stretch', 'colour', 'mag_stretch', 'mag_colour', 'stretch_colour']
        self.covs = {}

        ### Pantheon: has_mag_covmat = T --> read mag_covmat_file in full_long.dataset
        for name in covmats:
            if ini.bool('has_%s_covmat' % name):
                self.log.debug('Reading covmat for: %s ' % name)
                self.covs[name] = self._read_covmat(
                    os.path.join(self.path, ini.string('%s_covmat_file' % name)))
        
        ### Since only one covmat item ('mag')
        self.alphabeta_covmat = (len(self.covs.items()) > 1 or
                                 self.covs.get('mag', None) is None)

        self._last_alpha = np.inf
        self._last_beta = np.inf

        ### In jla_lite.yaml only
        self.marginalize = getattr(self, "marginalize", False)

        assert self.covs

        # If not sampling Mb, perform analytic marginalization
        if not self.use_abs_mag:
            zfacsq = 25.0 / np.log(10.0) ** 2
            self.pre_vars = self.mag_var + zfacsq * self.pecz ** 2 * (
                (1.0 + self.zCMB) / (self.zCMB * (1 + 0.5 * self.zCMB))) ** 2

        # Debugging likelihood
        if debug_post == True:
            self.count = 0
        
        ### Return inverse covmat
        self.inverse_covariance_matrix()

    def get_requirements(self):
        if self.require_theory:

            # State requisites to the theory code
            reqs = {"angular_diameter_distance": {"z": self.zCMB}}
            
            ### True for cosmosis
            if self.use_abs_mag:
                reqs["Mb"] = None

            return reqs

        #return reqs
        else:
            return {}

    def _read_covmat(self, filename):

        ################################ COSMOSIS ################################
        #filename = self.options.get_string("covmat_file", default=default_covmat_file)
        print("Loading Pantheon covariance from {}".format(filename))
        
        f = open(filename)
        line = f.readline()
        n = int(len(self.zCMB)) # reading selected zcmb
        C = np.zeros((n,n))
        ii = -1
        jj = -1
        mine = 999
        maxe = -999
        for i in range(self.origlen):
            jj = -1
            if self.cut[i]:
                ii += 1
            for j in range(self.origlen):
                if self.cut[j]:
                    jj += 1
                val = float(f.readline())
                if self.cut[i]:
                    if self.cut[j]:
                        C[ii,jj] = val
        f.close()

        return C

    def inverse_covariance_matrix(self, alpha=0, beta=0):
        ### Copy covmat with 'mag' label
        if 'mag' in self.covs:
            invcovmat = self.covs['mag'].copy()
        else:
            invcovmat = 0
        
        ### delta = diagonal part of the statistical uncertainty
        # delta = self.pre_vars
        # np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)

        ### Take inverse
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov

    def alpha_beta_logp(self, lumdists, alpha=0, beta=0, Mb=0, invcovmat=None):
        ### True for cosmosis
        if self.use_abs_mag:
            estimated_scriptm = Mb
        
        else:
            ### Inverse of systematic uncertainties
            invvars = 1.0 / self.pre_vars

            ### Sum of invvars
            wtval = np.sum(invvars)

            ### Estimated, instead of sampled, Mb
            estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval #+ 25
        
        ### Data vector? Diff between theory and observed, with correction
        ### (self.mag - estimated_scriptm) = observed/corrected mu?
        diffmag = self.mag - lumdists - estimated_scriptm
        
        ### Retreive invcov
        invcovmat = self.invcov

        ### Taking dot product to get chi^2
        invvars = invcovmat.dot(diffmag) # (covmat)^-1 dot datavector
        amarg_A = invvars.dot(diffmag)   # innvars dot data vector

        amarg_B = np.sum(invvars)
        amarg_E = np.sum(invcovmat)

        if self.use_abs_mag:
            chi2 = amarg_A + np.log(amarg_E / _twopi)
        
        ### Analytic marginalization?
        else:
            chi2 = amarg_A + np.log(amarg_E / _twopi) - amarg_B ** 2 / amarg_E
        
        ##
        #print("Cobaya Likelihood: ", -chi2/2)
        if record_like:
            file_like.write(str(-chi2/2)+ '\n')

        
        if debug_post == True: #chi2 == np.inf or chi2 == -np.inf:
            self.count += 1
            #file_lumdists.write("invvars: ")
            np.savetxt("%sinvvars_%d"%(root, self.count), invvars)
            np.savetxt("%sdiffmag_%d"%(root, self.count), diffmag)
            #file_lumdists.write(str(invvars.flatten()))
            #file_lumdists.write("\n")
        
            file_lumdists.write("amarg_A: %.4f \n"%amarg_A)  
            file_lumdists.write("\n")

        return - chi2 / 2

    def logp(self, **params_values):
        
        
        ### Calculate theory using class/camb
        if self.require_theory:
            angular_diameter_distances = \
                self.provider.get_angular_diameter_distance(self.zCMB)
            angular_diameter_distances = np.array(angular_diameter_distances) # Vectorize
            
            lumdists = np.array((5 * np.log10((1 + self.zHEL) * (1 + self.zCMB) *
                                    angular_diameter_distances)) + 25)

        else:
            # Calculating cosmological distances using manual integrator
            print("Warning: Not using CAMB or CLASS")
            H0 = params_values['H0']
            omegam = params_values['Omega_m']

            lumdists = []
            for  i in range(len(self.zCMB)):
                lumdists.append(da(self.zCMB[i], H0,omegam) * (1. + self.zCMB[i])*(1. + self.zHEL[i]))
            lumdists = np.array(lumdists) # Vectorize
            lumdists = 5 * np.log10(lumdists) + 25
        
        if debug_post == True:
            with open(file_lumdists_name, "w") as file_lumdists:
                file_lumdists.write("lumdist avg: %.3f \n"%np.average(lumdists))

        '''
        H0 = params_values['H0']
        omegam = params_values['Omega_m']

        # calculating cosmological distances
        angular_diameter_distances = \
                self.provider.get_angular_diameter_distance(self.zCMB)

        # Test: compute classy angular diameter distances and compare
        lumdists_class = (5 * np.log10((1 + self.zHEL) * (1 + self.zCMB) *
                                    angular_diameter_distances)) + 25

        lumdists = []
        for  i in range(len(self.zCMB)):
            lumdists.append(da(self.zCMB[i], H0,omegam) * (1. + self.zCMB[i])*(1. + self.zHEL[i]))
        lumdists = np.array(lumdists) # Vectorize
        lumdists = 5 * np.log10(lumdists) + 25

        lumdist_diff = np.abs(lumdists-lumdists_class)
        if record_theory:
            file_distances.write(str(lumdists) + " " + str(lumdists_class) + " " + str(lumdist_diff) + '\n')
        #print("Manual: ")
        #print(lumdists)
        #print("Class: ")
        #print(lumdists_class)
        #print("differences")
        #print(lumdist_diff)
        #print("manual: %.3f"%lumdists[-1])
        #print("class: %.3f"%lumdists_class[-1])
        #print("difference: %.3f"%lumdist_diff[-1])
        '''

        if self.use_shoes:
            # replacing distances to calibration SNe with distance measurements
            lumdists[self.is_calibrator] = self.cepheid_distance[self.is_calibrator]
       
        if self.use_abs_mag:
            Mb = params_values.get('Mb', None)
            #print(Mb)
            #print("DEBUG: Sampling Mb")

        else:
            print("DEBUG: Not sampling Mb")
            Mb = 0
            
        return self.alpha_beta_logp(lumdists, Mb=Mb)
