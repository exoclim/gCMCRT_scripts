import numpy as np
from netCDF4 import Dataset
import matplotlib.pylab as plt
import iris
import warnings
import aeolus.coord as c
from bs99 import get_mmr_bs99

warnings.filterwarnings("ignore",category=UserWarning,module="iris")

# TO DO
# -----
# * Allow it to specify specific time output
# * Convert to a function
# * Allow overriding of .conf specifications
# * Where is the antistellar/substellar point?
# * Are lat/lon midpoints or boundaries?
# * Add check in reduce_species_list for species that are only in the CIA list (e-)

outname = "UM"
outdir = "../UM_WASP39b_Test"
verbose = True
chem_type = 'FromOutputs'
param_type = 'User'
dumpnum = 990
alkali_fix = True
radius = 89416169.
R = 3164.6495
mdh = 1.

# Filename to import
um_dir = '/u/dchristie/WASP39b/'

# Opacity Switches
l_corr_k = True
l_lbl = False
l_rayleigh = True
l_cia = True
l_cloud = False
l_xsec = False

# Opacity sources, used to reduce the species list 
k_sources = ['C2H2','CO','FeII','HCl','K','OH','SiO','CH4','Fe','H2O','HCN','Na','PH3','TiO','CO2','FeH','H2S','HF','NH3','SH','VO']
rayleigh_sources = ['H2','He','H','e-']
cia_sources = ['H2-H2','He-H2','H2-H','H-','H2-','He-']
cia_files = ['HITRAN/H2-H2_2011.cia','HITRAN/H2-He_2011.cia','HITRAN/H2-H_2011.cia','','H2-_ff.txt','He-_ff.txt']
cia_formats = [4,4,4,4,0,2,2]

# Dictionary of UM chemical species, mapped to their STASH entries
STASH = {
    "m01s56i001": "H",
    "m01s56i002": "OH",
    "m01s56i003": "CH3",
    "m01s56i004": "H2CO",
    "m01s56i005": "O-3P",
    "m01s56i006": "NH2",
    "m01s56i007": "H2",
    "m01s56i008": "CO",
    "m01s56i009": "H2O",
    "m01s56i010": "HCO",
    "m01s56i011": "NH",
    "m01s56i012": "HCN",
    "m01s56i013": "CH3OH",
    "m01s56i014": "CH4",
    "m01s56i015": "CH3O",
    "m01s56i016": "NCO",
    "m01s56i017": "CH2OH",
    "m01s56i018": "N2H2",
    "m01s56i019": "NH3",
    "m01s56i020": "NNH",
    "m01s56i021": "CN",
    "m01s56i022": "HNCO",
    "m01s56i023": "3CH2",
    "m01s56i024": "N2",
    "m01s56i025": "N2H3",
    "m01s56i026": "1CH2",
    "m01s56i027": "HOCN",
    "m01s56i028": "H2CN",
    "m01s56i029": "CO2",
    "m01s56i030": "He",
}

# Directories for opacity data
k_dir = "../data/k_tables/R100_ck_data_g16_UV"
cia_dir = "../data/CIA_tables/"

def reduce_species_list(spec_in):
    spec_out = []
    spec_k = []
    spec_ray = []
    for s in spec_in:
        if ((s in k_sources) or (s in rayleigh_sources)):
            spec_out.append(s)
        if (s in k_sources):
            spec_k.append(s)
        if (s in rayleigh_sources):
            spec_ray.append(s)
        
    return(spec_out,spec_k,spec_ray)

def reduce_cia_list(spec_in):
    spec_cia = []

    for l in cia_sources:
        s1,s2 = l.split("-")
        if (s2 == ""):
            if ((s1 in spec_in) and ('e-' in spec_in)):
                spec_cia.append(l)
        else:
            if ( (s1 in spec_in) and (s2 in spec_in) ):
                spec_cia.append(l)
    return(spec_cia)


# Extract relevant data from the rose-app-run.conf file.

# TODO: What units do these need to be in? Currently in SI

if (param_type == "Import"):
    f = open(um_dir + "/rose-app-run.conf","r")

    for l in f.readlines():
        if ("=" in l and not l.startswith("!!")):
            s = l.split("=")
            key = s[0]
            val = s[1]

            if (key == "pref"):
                p_ref = float(val)
            if (key == "cp"):
                cp = float(val)
            if (key == "planet_radius"):
                radius = float(val)
            if (key == "g"):
                g = float(val)
            if (key == "r"):
                R = float(val)

    f.close()

    if (verbose):
        print("Imported simulation parameters:")
        print("    p_ref  = ",p_ref)
        print("    cp     = ",cp)
        print("    radius = ",radius)
        print("    R      = ",R)
        print("    g      = ",g)

else:
     # Just assume that the parameters are appropriately specified.
    if (radius <= 0.):
        print("Error: radius is not positive.")
        exit()
    if (R <= 0.):
        print("Error: R is not positive.")
        exit()
    if (verbose):
        print("Specified simulation parameters:")
        print("    radius = ",radius)
        print("    R      = ",R)
        
        
# Compute mu, converted to g/mol
mu = 8.31446261815324/R*1000.
if (verbose):
    print("Computed parameters: ")
    print("    mu     = ",mu," g/mol")
    
# Open the dump in iris
files = [um_dir + "/atmosa.p{}000000{:02}_00".format(x,dumpnum) for x in ["b","c","d","e","f","g","h"]]
fields = ['air_pressure','air_temperature','x_wind','y_wind','upward_air_velocity']

if (verbose):
    print("Loading cubes.")
cubes = iris.load(files,fields)
pcube_old = cubes[0]
tcube = cubes[1]

# Read wind and remap onto the pressure grid
ucube_old = cubes[2]
vcube_old = cubes[3]
wcube = cubes[4]

pcube = c.regrid_3d(pcube_old,tcube)
ucube = c.regrid_3d(ucube_old,tcube)
vcube = c.regrid_3d(vcube_old,tcube)

lay = tcube.coord('level_height').points
lev = np.concatenate((np.array([-pcube.coord('level_height').bounds[0,1]]),pcube.coord('level_height').bounds[:,1]))
lons = tcube.coord('longitude').points
lats = np.array(tcube.coord('latitude').points)

it = -1

# Get dimensions of system
#nt = len(time)
nlat = len(lats)
nlon = len(lons)
nlay = len(lay)
nlev = nlay + 1
ni = nlay * nlat * nlon

# Shift the latitude coordinate to start from 0-180, the gCMCRT coordinate system
lats[:] = lats[:] + 90.0

if (verbose):
    print("Grid shape:")
    print('    nlats:  ', nlat)
    print('    nlons:  ', nlon)
    print('    nlay:   ', nlay)



# Extract temperature and presssure
T = tcube.data[:,:,:]
P = pcube.data[:,:,:]

# Extract winds
U = ucube.data[:,:,:]
V = vcube.data[:,:,:]
W = wcube.data[:,:,:] 

# Now we've got to shift longitudes to make the 0th index start at 0 degrees
# gCMCRT works from 0-360 degree longitude coordinates

#nr = int(nlon/2)
#lons[:] = np.roll(lons[:],nr)
#lons[:] = np.where(lons[:] > 0, lons[:], 360.0-abs(lons[:]))

#print(lons[:])

# Chemistry

if chem_type == "BS99":
    if (verbose):
        print("Burrows and Sharp 1999 chemistry being used.")
    # Species in the GCM output file
    sp_bs99 = ['H2','H2O','CO','CH4','Na','K','NH3','He']
    sp,sp_k,sp_ray = reduce_species_list(sp_bs99)
    nsp = len(sp)
    VMR = np.zeros((nlay,nlat,nlon,nsp))

    # Compute the volume mixing ratios for each species
    if (verbose):
        print("Computing VMRs")
    for j in range(nlat):
        for i in range(nlon):
            for k in range(nlay):
                for n in range(nsp):
                    VMR[k,j,i,n] = get_mmr_bs99(T[k,j,i],P[k,j,i],sp[n],mdh=mdh)
        
elif chem_type == "FromOutputs":
    if (verbose):
        print ("VMRs being taken from UM outputs.")

    chem_cubes = iris.load(files,iris.AttributeConstraint(STASH=lambda stash: str(stash).startswith('m01s56i')))
    sp_list = []
    for c in chem_cubes:
        spec_name = STASH[str(c.attributes['STASH'])]
        sp_list.append(spec_name)
    print("Species in Stash: ",sp_list)

    if (alkali_fix):
        # First check that alkali metals aren't in the species list
        if ('Na' in sp_list or 'K' in sp_list):
            print("Error: Na/K already in the STASH.  The fix might no be needed.")
            exit
        sp_list.append("Na")
        sp_list.append("K")
        print("Added Na and K to the list. Will use BS99 VMR estimate.")
    
    sp,sp_k,sp_ray = reduce_species_list(sp_list)
    print("Species used in gCMCRT opacity calculation: ",sp)
    nsp = len(sp)
    VMR = np.zeros((nlay,nlat,nlon,nsp))
    i = 0
    for c in chem_cubes:
        if (STASH[str(c.attributes['STASH'])] in sp):
            if (STASH[str(c.attributes['STASH'])] != sp[i]):
                print("Something is wrong.")
                exit
            VMR[:,:,:,i] = c.data[:,:,:]
            i = i + 1
    if (alkali_fix):
        if (sp[i] != "Na" or sp[i+1] != "K"):
            print("Something is very wrong.")
            exit
        for j in range(nlat):
            for m in range(nlon):
                for k in range(nlay):
                    VMR[k,j,m,i] = get_mmr_bs99(T[k,j,m],P[k,j,m],"Na",mdh=mdh)
                    VMR[k,j,m,i+1] = get_mmr_bs99(T[k,j,m],P[k,j,m],"K",mdh=mdh)
else:
    print("Unknown chem_type.")
    exit()

# Header
head = open('../data/header.txt','r')
lines = head.readlines()
    
fname = outdir + '/{}.prf'.format(outname)
print('Outputting main profile: ', fname)
prf = open(fname,'w')
prf.write(lines[0])
prf.write(lines[1])
prf.write(str(ni) + '\n')
prf.write(lines[2])
prf.write(str(nsp) + '\n')
for n in range(nsp):
    prf.write(sp[n] + '\n')
prf.write(lines[3])
prf.write(lines[4])
n = 0
for k in range(nlat):
    for j in range(nlon):
        for i in range(nlay):
            prf.write(str(n+1) + ' ' + str(P[i,k,j]/1e5) + ' ' + str(T[i,k,j]) + ' ' + str(mu) + ' ' + " ".join(str(l) for l in VMR[i,k,j,:]) + '\n')
            n = n + 1
prf.close()

# Now output T-p profile in bar and K to an interpolation file (iprf) - after which we can interpolate to GGChem values to get VMRs
# Pressure units: bar
# T units: K
fname = outdir + '/{}.iprf'.format(outname)
if(verbose):
    print('Outputting interpolatable T-p profile: ', fname)
f = open(fname,'w')
n = 0
for j in range(nlat):
    for i in range(nlon):
        for k in range(nlay):
            f.write(str(n+1) + ' ' +  str(P[k,j,i]/1e5) + ' ' + str(T[k,j,i]) + '\n')
            n = n + 1
f.close()

# Now output Height grid in cm
fname = outdir + '/{}.hprf'.format(outname)
if(verbose):
    print('Outputting height profile: ', fname)
f = open(fname,'w')
for k in range(nlev):
  f.write(str(k+1) + ' ' + str((radius + lev[k]) * 100.0) + '\n')
f.close()


#Output the 3D wind profiles in CMCRT layout -> (lat,lon,lay) in units cm s-1
fname = outdir +'/{}.wprf'.format(outname)
if(verbose):
    print('Outputting wind profile: ', fname)
f = open(fname,'w')
n = 0
for j in range(nlat):
    for i in range(nlon):
        for k in range(nlay):
            f.write(str(n+1) + ' ' +  str(U[k,j,i] * 100.0) + ' ' + str(V[k,j,i] * 100.0) + ' ' + str(W[k,j,i] * 100.0)  + '\n')
            n = n + 1
f.close()

# Generate the optools parameter file

fname = outdir  + "/optools.par"
if (verbose):
    print("Writing optools.par file.")
f = open(fname,"w")
f.write("! Parameter file for optools routines\n")
f.write('!\n')
f.write('! ---- prf name file to process  ---- !\n')
f.write('! (Note, do not include the .prf extension)\n')
f.write("{}\n".format(outname))
f.write('!\n')
f.write('! ---- Main Switches ---- !\n')
f.write('!\n')
if (l_corr_k):
    f.write(".True.\n")  
else:
    f.write(".False.\n")
if (l_lbl):
    f.write(".True.\n")
else:
    f.write(".False.\n")
if (l_cia):
    f.write(".True.\n")
else:
    f.write(".False.\n")
if (l_rayleigh):
    f.write(".True.\n")
else:
    f.write(".False.\n")
if (l_cloud):
    print("Warning: I'm not sure this script sets up cloud opacity properly.")
    f.write(".True.\n")
else:
    f.write(".False.\n")
if (l_xsec):
    f.write('.True.\n')
else:
    f.write('.False.\n')
f.write("!\n")
f.write("! ---- Gas Opacity Properties ---- !\n")
f.write("! ! Number of corr-k line opacity species followed by names\n")
f.write("{}\n".format(len(sp_k)))
for i in range(len(sp_k)):
    f.write("{}\n".format(sp_k[i]))
f.write("!\n")
f.write("! ---- Gas Opacity Properties ---- !\n")
f.write("! ! Number of lbl line opacity species followed by names\n")
f.write("0\n") # LBL goes here
f.write("!\n")
f.write("! ---- Gas Rayleigh scattering  ---- !\n")
f.write("! ! Number of Rayleigh scattering gases followed by names\n")
f.write("{}\n".format(len(sp_ray)))
for i in range(len(sp_ray)):
    f.write("{}\n".format(sp_ray[i]))
f.write("!\n")
f.write("! ---- Gas Continuum opacity  ---- !\n")
f.write("! ! Number of Continuum gases followed by names\n")

sp_cia = reduce_cia_list(sp)
f.write("{}\n".format(len(sp_cia)))
for l in sp_cia:
    f.write("{}\n".format(l))
f.write("!\n")
f.write("! ---- Cloud Species  ---- !\n")
f.write("! ! Number of cloud species followed by names\n")
f.write("0\n")
f.write('!\n')
# xsec species
f.write('! ---- xsec Species  ---- !\n')
f.write('! ! Number of xsec species followed by names\n')
f.write('0\n')
f.close()

# Write the namelist
fname = outdir  + "/optools.nml"
if (verbose):
    print("Writing optools.nml file.")
f = open(fname,"w")

f.write("! Fortran namelist for optools\n\n")

if (verbose):
    print("....CK_nml namelist.")

f.write("&CK_nml\n")

f.write("pre_mixed = .False.\n")
f.write("interp_wl = .False.\n")  # Think about this
f.write("iopts = 1\n")
f.write("form = {}\n".format(",".join( ['2' for i in range(0,len(sp_k))])))
f.write("nG = 16\n")
f.write("paths = {}\n".format(",\n".join(["'" + k_dir + "/{}_ck_HELIOSK_R100.txt'".format(x) for x in sp_k]))) 
f.write("/\n")

if (verbose):
    print("....lbl_nml namelist. Currently unsupported in this script.")

# Line by line - not used but included
f.write("\n&lbl_nml\n")

f.write("interp_wl = .False.\n")
f.write("iopts = 1\n")
f.write("form = 0\n")
f.write("paths = ''\n")
f.write("/\n")

# CIA
if (verbose):
    print("....CIA_nml namelist.")
f.write("&CIA_nml\n")
f.write("iopts = 1\n")

# Construct the file list
form_list = []
path_list = []
for l in sp_cia:
    idx = cia_sources.index(l)
    form_list.append(str(cia_formats[idx]))
    path_list.append("'" + cia_dir + cia_files[idx] + "'")
f.write("form = {}\n".format(",".join(form_list)))
f.write("paths = {}\n".format(",\n".join(path_list))) 
f.write("/\n\n")


#paths = '../data/CIA_tables/HITRAN/H2-H2_2011.cia',
#'../data/CIA_tables/HITRAN/H2-He_2011.cia',
#'../data/CIA_tables/HITRAN/H2-H_2011.cia',
#'../data/CIA_tables/HITRAN/He-H_2011.cia',
#'',
#'../data/CIA_tables/H2-_ff.txt'
#'../data/CIA_tables/He-_ff.txt'
#/

# Clouds - Not implemented yet
if (verbose):
    print("....Rayleigh_nml namelist.")

f.write("&Rayleigh_nml\n")
f.write("iopts = 1\n")
f.write("/\n\n")

if (verbose):
    print("....cl_nml namelist. Currently unsupported in this script.")

f.write("&cl_nml\n")

f.write("iopts = 1\n")
f.write("imix = 1\n")
f.write("idist = 3\n")
f.write("ndist = 101\n")
f.write("idist_int = 1\n")
f.write("imie = 1\n")
f.write("form = 5\n")
f.write("paths = '../data/nk_tables/Al2O3[s].dat'\n")
f.write("sig = 1.0\n")
f.write("eff_fac = 0.125\n")
f.write("veff = 0.125\n")
f.write("amin = 1.0e-2\n")
f.write("amax = 100.0\n")
f.write("fmax = 0.95\n")
f.write("/\n\n")
f.close()
