import numpy as np

# Constants
kB = 1.38e-23

# Parameters for the removal of Oxygen
T_remove_O=1500.
# Characteristic transition scale in T over which O is removed
T_trans_scale_remove_O=10.0

# Characteristic transition scale in T over which the abundance changes                                  
T_trans_scale_TiO = 10.0                                            
T_trans_scale_VO  = 10.0                                            
T_trans_scale_Na  = 20.0                                            
T_trans_scale_K   = 20.0                                            
T_trans_scale_Li  = 20.0                                            
T_trans_scale_Rb  = 20.0                                            
T_trans_scale_Cs  = 20.0

# Elemental abundances as log(epsilon_z) = log(N_z/N_H) + 12, where N_z is the
# number of atoms of species z
log_eps_C  = 8.50                                                  
log_eps_N  = 7.86                                                   
log_eps_O  = 8.76                                                   
log_eps_Na = 6.24                                                   
log_eps_Si = 7.51                                                   
log_eps_K  = 5.03                                                   
log_eps_Ti = 4.95                                                   
log_eps_V  = 3.93                                                   
log_eps_Li = 3.26                                                   
log_eps_Rb = 2.52                                                   
log_eps_Cs = 1.08

# Convert elemental abundances to number of atoms relative to H:
# APrim_z = N_z/N_H
APrim_C  = (10.0)**(log_eps_C  - 12.0)                        
APrim_N  = (10.0)**(log_eps_N  - 12.0)                        
APrim_O  = (10.0)**(log_eps_O  - 12.0)                        
APrim_Na = (10.0)**(log_eps_Na - 12.0)                        
APrim_Si = (10.0)**(log_eps_Si - 12.0)                        
APrim_K  = (10.0)**(log_eps_K  - 12.0)                        
APrim_Ti = (10.0)**(log_eps_Ti - 12.0)                        
APrim_V  = (10.0)**(log_eps_V  - 12.0)                        
APrim_Li = (10.0)**(log_eps_Li - 12.0)                        
APrim_Rb = (10.0)**(log_eps_Rb - 12.0)                        
APrim_Cs = (10.0)**(log_eps_Cs - 12.0)

# Equilibrium coefficients from Burrows & Sharp, ApJ, 1999:
                                                    
eqcoeff_1 = [                                                          
         1.106131e+06,                                                   
        -5.6895e+04,                                                     
         62.565,                                                         
        -5.81396e-04,                                                    
         2.346515e-08]                                                
eqcoeff_2 = [                                                          
         8.16413e+05,                                                    
        -2.9109e+04,                                                     
         58.5878,                                                        
        -7.8284e-04,                                                     
         4.729048e-08]

# Molar weights from SOCRATES
molar_weight = {'H2O':18.0153,   
  'CO2':44.0100, 
  'O3':47.9982,  
  'N2O':44.0128,   
  'CO':28.0106,   
  'CH4':16.0430,  
  'O2':31.9988,  
  'NO':30.0061,  
  'SO2':64.0628,   
  'NO2':46.0055,  
  'NH3':17.0306,   
  'HNO3':63.0129,   
  'N2':28.0134,  
  'CFC11':137.3686,  
  'CFC12':120.9140,  
  'CFC13':187.3765,  
  'HCF22':86.46892,  
  'HFC125':120.02227,  
  'HFC134a':102.03184,   
  'CFC114':170.921,   
  'TiO':63.866,    
  'VO':66.9409,   
  'H2':2.01588,   
  'He':4.002602,  
  'OCS':60.075,   
  'Na':22.98976928, 
  'K':39.0983,  
  'FeH':56.853,  
  'CrH':53.004,   
  'Li':6.941,     
  'Rb':85.4678,  
  'Cs':132.9054519, 
  'PH3':33.99758,    
  'C2H2':26.0373,   
  'HCN':27.0253,  
  'H2S':34.081,  
  'Ar':39.948}  

def calc_temp_curve(P,spec):
	poly_highp_tio_vo =  [-3.96274342e-05, 5.20741797e-04]                                                  
	poly_highp_na = [ -5.58256919e-05, 8.81851644e-04]                                                  
	poly_highp_k = [ -5.46977180e-05, 8.19104478e-04 ]                                       
	poly_highp_li = [ -3.50995394e-05,6.51993843e-04 ]                                             
	poly_highp_rb = [ -6.06654087e-05,8.09569974e-04 ]
	poly_highp_cs = [ -5.29210264e-05,7.71577097e-04]


	poly_lowp_tio_vo  = [ -2.91788901e-05,  5.11801742e-04] 
	poly_lowp_na = [ -6.69921629e-05, 8.90116829e-04] 
	poly_lowp_k = [-6.46633572e-05, 8.29549449e-04]                                               
	poly_lowp_li = [-3.55469185e-05, 6.52116945e-04 ]                                                  
	poly_lowp_rb = [ -3.19328287e-05,  8.69542964e-04 ]
	poly_lowp_cs = [ -3.85306167e-05,7.63040762e-04 ]


	# Log (base 10) of transition point between polynomial fits                                                   &
	log_p_trans_tio_vo =  1.0                                             
	log_p_trans_na =      1.0                                             
	log_p_trans_k =       1.0                                             
	log_p_trans_li =      1.0                                             
	log_p_trans_rb =     -2.0                                             
	log_p_trans_cs =      1.0

	if ((spec == 'TiO') or (spec == 'VO')):
		poly_highp = poly_highp_tio_vo
		poly_lowp = poly_lowp_tio_vo
		log_p_trans = log_p_trans_tio_vo
	elif (spec == 'Na'):
		poly_highp = poly_highp_na
		poly_lowp = poly_lowp_na
		log_p_trans = log_p_trans_na
	elif (spec == 'K'):
		poly_highp = poly_highp_k
		poly_lowp = poly_lowp_k
		log_p_trans = log_p_trans_k
	elif (spec == 'Li'):
		poly_highp = poly_highp_li
		poly_lowp = poly_lowp_li
		log_p_trans = log_p_trans_li
	elif (spec == 'Rb'):
		poly_highp = poly_highp_rb
		poly_lowp = poly_lowp_rb
		log_p_trans = log_p_trans_rb
	elif (spec == 'Cs'):
		poly_highp = poly_highp_cs
		poly_lowp = poly_lowp_cs
		log_p_trans = log_p_trans_cs

	log_p_bar = np.log10(P/1.0e+05)

	work = 1.0 / (np.exp(-(log_p_bar-log_p_trans)/0.1) + 1.0)
	t_cond = 1.0 / ( (poly_lowp[0]*log_p_bar + poly_lowp[1])) * (1.0 - work) +  1.0 / ( (poly_highp[0]*log_p_bar + poly_highp[1])) * work
	return t_cond

def get_mmr_bs99(T,P,spec,mdh=0.):

	r_gas = 8.3143e+00

	a_h  = 0.91183e+00
	A_He = 1.0e+00 - a_h

	# Ideal gas constant in cal/(mol*K)
	r_gas_cal=1.9858775

	# Number of atmospheres per Pa
	atmprpa=1.0/1.01325e+05

	# Average number of oxygen atoms removed per silicon atom
	x_Si=3.28

	# Calculates H2 and He partial pressures
	def calc_H2_He_pp(P):

		# Calculate partial pressures (weighing total pressure by abundances,
		# assume all H is in H2)
	    pp_H2 = P*(a_h/2.0)/(a_h/2.0 + A_He)
	    pp_He = P*(A_He)/(a_h/2.0 + A_He)

	    return pp_H2,pp_He

	def calc_K1(T):
		k = np.exp( (eqcoeff_1[0]/T + eqcoeff_1[1]                          
	      + eqcoeff_1[2]*T                                                   
	      + eqcoeff_1[3]*(T**2)                                                 
	      + eqcoeff_1[4]*(T**3))                                                
	      / (r_gas_cal*T ))
		return k

	def calc_K2(T):
		k = np.exp( (eqcoeff_2[0]/T + eqcoeff_2[1]                          
	      + eqcoeff_2[2]*T                                                    
	      + eqcoeff_2[3]*T**2                                                 
	      + eqcoeff_2[4]*T**3)                                                
	      / (r_gas_cal*T) )
		return(k)


	# Remove Oxygen

	frac_remove_O = 1.0 /(np.exp((T - T_remove_O)/T_trans_scale_remove_O) + 1.0)

	pp_H2, pp_He = calc_H2_He_pp(P)

	rho_tot = molar_weight['H2']*1.0e-03*pp_H2/(r_gas*T) + molar_weight['He']*1.0e-03*pp_He/(r_gas*T)

	if(spec == 'H2'):
		mmr = 1.0

	elif(spec == 'He'):
		mmr = pp_He/pp_H2

	elif( (spec == 'CO') or (spec == 'CH4') or (spec == 'H2O')):
		K1 = calc_K1(T)
		APrim_O_depleted = APrim_O - x_Si*APrim_Si*frac_remove_O
		CO_work = APrim_C + APrim_O_depleted + (pp_H2*atmprpa)**2 / (2.0*K1)
		mmr = CO_work - np.sqrt( CO_work**2 - 4.0*APrim_C*APrim_O_depleted)

		if (spec == 'CH4'):
			mmr = 2.0*APrim_C - mmr
		if (spec == 'H2O'):
			mmr = 2.0*APrim_O_depleted - mmr

	elif (spec == 'NH3'):
		K2 = calc_K2(T)
		NH3_work = (pp_H2*atmprpa)**2/(8.0*K2)

		if ( APrim_N / NH3_work < 1.0e-10):
			mmr = 2.0*APrim_N
		else:
			mmr = 2.0*( np.sqrt( NH3_work*(2.0*APrim_N + NH3_work) ) - NH3_work )

	elif (spec == "TiO"):
		T_cond = calc_temp_curve(P,spec)
		mmr = 2.0*APrim_Ti/(np.exp(-(T - T_cond)/T_trans_scale_TiO) + 1.0)

	elif (spec == "VO"):
		T_cond = calc_temp_curve(P,spec)
		mmr = 2.0*APrim_V/(np.exp(-(T - T_cond)/T_trans_scale_VO) + 1.0)

	elif (spec == "Na"):
		T_cond = calc_temp_curve(P,spec)
		mmr = 10.**(mdh)*2.0*APrim_Na/(np.exp(-(T - T_cond)/T_trans_scale_Na) + 1.0)

	elif (spec == "K"):
		T_cond = calc_temp_curve(P,spec)
		mmr = (10.0*mdh)*2.0*APrim_K/(np.exp(-(T - T_cond)/T_trans_scale_K) + 1.0)

	elif (spec == "Li"):
		T_cond = calc_temp_curve(P,spec)
		mmr = 2.0*APrim_Li/(np.exp(-(T - T_cond)/T_trans_scale_Li) + 1.0)


	elif (spec == "Rb"):
		T_cond = calc_temp_curve(P,spec)
		mmr = 2.0*APrim_Rb/(np.exp(-(T - T_cond)/T_trans_scale_Rb) + 1.0)

	elif (spec == "Cs"):
		T_cond = calc_temp_curve(P,spec)
		mmr = 2.0*APrim_Cs/(np.exp(-(T - T_cond)/T_trans_scale_Cs) + 1.0)
	else:
		print("Unsupported species: {}".format(spec))
                
	mmr = mmr *pp_H2/P

	return mmr


def get_mmr_bs99_profile(T,P,spec):

	nz = T.shape[0]

	mmr = np.zeros(nz)

	for i in range(0,nz):
		mmr[i] = get_mmr_bs99(T[i],P[i],spec)
	return mmr

def get_cond_curve_profile(P,spec):
	nz = P.shape[0]

	T_cond = np.zeros(nz)
	for i in range(0,nz):
		T_cond[i] = calc_temp_curve(P[i],spec)

	return T_cond

