import numpy as np

#####################################
############# some info #############
### In smilei, c = 1, mu_0 = 1, epsilon_0 = 1
### me = 1, mi = mass_ratio.
### electronic charge, e = 1, n0 = 1
### electron plasma frequency, omega_pe = sqrt((n0*e^2)/(me*epsilon_0)) = 1
### di = ion inertial length = sqrt(mass_ratio)*de, de = electron intertial length = c/omega_pe = 1
### Hence, di = sqrt(mass_ratio)
#####################################



###-----****-----****-----###****###-----****-----****-----###****###-----****-----****-----###
###-----****-----### Simulation parameters that the code requires ###-----****-----****-----###
###-----****-----****-----###****###-----****-----****-----###****###-----****-----****-----###
mass_ratio  = 100                             ##m_i/m_e
di          = np.sqrt(mass_ratio)           ## di = np.sqrt(mass_ratio)*de, de = 1.0 in Smilei
ion_mass    = mass_ratio
Delta       = 2.0                         ## width of the CS in terms of di
lamb_i      = Delta*di
Lx          = 6.4*lamb_i
dx	    = 0.00625*lamb_i
Nx          = int(Lx/dx)
dy          = dx

aspect_ratio= 1.0/2.0 	                     ## by definition: Lx/Ly = aspect_ratio; for Ly>Lx: aspect_ratio < 1.0
Ly          = Lx/aspect_ratio                ## By definition. 
Ny          = int(Ly//dy)
dt_CFL      = (dx**-2+dy**-2)**-0.5
dt          = 0.95 * dt_CFL
max_time    = 200*lamb_i                          ## simulation time
every_diag  = int(max_time/dt/200)                      ##int(max_time/dt/100)
part_num    = 128                            ## number of particles

###-----****-----****-----###****###-----****-----****-----###****###-----****-----****-----###
###-----****-----###  Physical parameters that the code requires  ###-----****-----****-----###
###-----****-----****-----###****###-----****-----****-----###****###-----****-----****-----###

R       = 0.5                           ## Asymmetry in B-field
alpha   = 0.45                          ## asymmetry in density
eta_CA  = 10.0 # np.sqrt(mass_ratio)                          ## eta_CA = c/VA = omega_pi/Omega_i
tau     = 1.0                           ## temp_ratio i.e. Ti/Te NOT in theta
r 	= 1.0/4.0
n0      = 1.0                           ## number density
B0      = (1.0/eta_CA)*(np.sqrt(mass_ratio*n0))                 ## equilibrium magnetic field
theta_e = (R/alpha)*(1.0/(1.0+tau))*(mass_ratio/(eta_CA**2))    ## initial electron temperature in dimensionless unit
theta_i = (tau)*theta_e                                         ## initial ion temperature in m_e*c**2
theta_e_neq = r*theta_e
theta_i_neq = r*theta_i
#V0       = -(1.0/(2.0*Delta*eta_CA))    ## peak drift velocity of the species

######################################
#####            Main            #####
######################################
Main(
        geometry = "2Dcartesian",
        number_of_cells = [Nx, Ny],
        cell_length = [dx, dy],
        simulation_time = max_time,
        #number_of_timesteps = max_time,
        timestep = dt,
        #       number_of_patches = [Nx/8,Ny/8],
        number_of_patches = [int(Nx/8), int(Ny/8)],
        # maxwell_solver = 'Lehe',
        # maxwell_solver = 'Yee',
        interpolation_order = 2,
        EM_boundary_conditions = [["reflective","silver-muller"],['periodic']],
        # EM_boundary_conditions = [["periodic"],["silver-muller"]],
        # EM_boundary_conditions = [["silver-muller"]],
        print_every = every_diag
 )

###-----****-----****-----###****###-----****-----****-----###****###-----****-----****-----###
###-----****-----###   Some definitions that the code requires    ###-----****-----****-----###
###-----****-----****-----###****###-----****-----****-----###****###-----****-----****-----###
sech = lambda x: 1 / np.cosh(x)

### nember density variation
def n(x,y):
    X = x-Lx/2
    Y = y-Ly/2
    return n0*(1.0-alpha*np.tanh((X-0.25*Lx)/lamb_i)-alpha*np.tanh((X-0.25*Lx)/lamb_i)**2 )

####################
## Magnetic field ##
####################
def Bx(x,y):
    return 0.0

def By(x,y):
    X = x-Lx/2
    return B0*(np.tanh((X-0.25*Lx)/lamb_i)+ R)

def Bz(x,y):
    return 0.0

#####################
## current density ##
#####################
def Jz(x,y):
    X = x-Lx/2
    return (B0/lamb_i)*(sech((X-0.25*Lx)/lamb_i)**2)

###############################################################
###                   The drift velocity                    ###      
### The logic of the next three definitions is following.   ###
### Vz_ie = Vz_i - Vz_e = Jz/(n*e)                          ###
### Harris frame: Vz_i = -Vz_e                              ###
###############################################################

def Vz_ie(x,y):
    return Jz(x,y)/n(x,y)
def Vz_i(x,y):
    return Vz_ie(x,y)/2
def Vz_e(x,y):
    return -Vz_i(x,y)

####################
LoadBalancing()
Vectorization(mode="on")


def ppc(x,y):
    return part_num*(n(x,y)/n0)
###################################################
###             defining species                ###
###################################################
Species(
        name = 'eons',
        position_initialization = 'random',
        momentum_initialization = 'maxwell-juettner',
        particles_per_cell = ppc,
        mass = 1,
        number_density = n,
        charge = -1,
        mean_velocity = [0,0,Vz_e],
        #temperature = [theta_e,theta_e,theta_e],
        temperature = [theta_e_neq,theta_e_neq,theta_e_neq],
        boundary_conditions = [['reflective','remove'],['periodic']],
        # boundary_conditions = [['remove']],
        )

Species(
        name = 'ions',
        position_initialization = 'random',
        momentum_initialization = 'maxwell-juettner',
        particles_per_cell = ppc,
        mass = ion_mass,
        number_density = n,
        charge = 1,
        mean_velocity = [0,0,Vz_i],
        #temperature = [theta_i,theta_i,theta_i],
        temperature = [theta_i_neq,theta_i_neq,theta_i_neq],
	boundary_conditions = [['reflective','remove'], ['periodic']],
        # boundary_conditions = [['remove']],

        )

###########################################
### injecting species from x-boundaries ###
###########################################

defined_species = ['eons', 'ions' ]

for part_species in defined_species:
    # for boundary in ['xmin', 'xmax']:
    for boundary in ['xmax']:
        ParticleInjector(
                            species = part_species,
                            box_side = boundary
                        )


##############################################
### providing the external magnetic ifelds ###
##############################################

ExternalField(
                field = 'Bx',
                profile = Bx,
             )

ExternalField(
                field = 'By',
                profile = By,
             )

ExternalField(
                field = 'Bz',
                profile = Bz,
             )

########################
##### Diagnostics ######
########################

#Nt=int(0/dt)
#every1=[Nt, every_diag]
save_every = 1

DiagScalar(
            every = every_diag,
          )

DiagFields(
        #   name = "/diag/diag1",
            every = every_diag,
            time_average = 1,
            fields = ['Bx','By','Bz','Ex','Ey','Ez','Jx','Jy','Jz'],
          )

for species in ["eons","ions"]:
    for quantity in ["weight","weight_px",'weight_py','weight_pz',
                     "weight_charge","weight_charge_vx","weight_charge_vy","weight_charge_vz", "weight_ekin",
                        "weight_vx_px","weight_vy_py","weight_vz_pz",
                        "weight_vx_py","weight_vy_pz","weight_vx_pz"]:
        DiagParticleBinning(
            deposited_quantity = quantity,
            every = every_diag,
            time_average = save_every,
            species = [species],
            axes = [["x",0,Lx,Nx+1],
                    ["y",0,Ly,Ny+1]],
        )
