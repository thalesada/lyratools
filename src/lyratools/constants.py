"""Physical constants (cgs), unit conversions and LYRA conventions used throughout lyratools."""

# physical constants, cgs
G = 6.6743e-8                 # cm^3 g^-1 s^-2
PROTONMASS = 1.67262192e-24   # g
BOLTZMANN = 1.380649e-16      # erg / K
MSUN = 1.98841e33             # g
PC = 3.0856776e18             # cm
KPC = 1e3 * PC
YR = 3.15576e7                # s
MYR = 1e6 * YR
GYR = 1e9 * YR
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0

# gravitational constant in astronomer units
G_KPC_KMS2_PER_MSUN = 4.30091e-6   # kpc (km/s)^2 / Msun
G_PC_KMS2_PER_MSUN = 4.30091e-3    # pc (km/s)^2 / Msun

# rate conversions
KMS_PER_KPC_TO_PER_YR = 1.0227122e-9   # (km/s)/kpc -> 1/yr
PC_PER_KMS_TO_MYR = 0.9777922          # pc/(km/s) -> Myr

# composition
HYDROGEN_MASSFRAC = 0.76
ZSOLAR = 0.0133697                     # solar metal mass fraction used in LYRA analyses
MU_IONIZED = 4.0 / (1.0 + 3.0 * HYDROGEN_MASSFRAC)

# order of species in LYRA's GFM_Metals arrays
ELEMENTS = {"H": 0, "He": 1, "C": 2, "N": 3, "O": 4, "Mg": 5, "Si": 6, "Mn": 7, "Fe": 8, "Other": 9}
# row of each element in the Asplund et al. (2009) solar composition table
SOLAR_TABLE_ROWS = {"H": 0, "He": 1, "C": 5, "N": 6, "O": 7, "Mg": 11, "Si": 13, "Mn": 24, "Fe": 25}

# LYRA SNTime convention (cosmological runs):
#   -1  : star never explodes as a SN
#   > 0 : SN still to come
#   < 0 : SN has exploded; |SNTime| is the stellar lifetime
SNTIME_NONE = -1.0
SNTIME_UNIT_MYR = 1e3          # |SNTime| is stored in Gyr
SN_MIN_MASS = 8.0              # Msun, minimum initial mass of a core-collapse SN progenitor

# AREPO code units (used only where no snapshot is available, e.g. parsing log files)
CODE_UNITS_COSMOLOGICAL = dict(length_cm=3.08568e24, mass_g=1.989e43, velocity_cms=1e5)
CODE_UNITS_ISOLATED = dict(length_cm=3.08568e21, mass_g=1.989e43, velocity_cms=1e5)
