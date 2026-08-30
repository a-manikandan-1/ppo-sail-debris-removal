"""
    Common constants in orbital mechanics
"""

EARTH_RADIUS = 6.378E+06 # radius of the Earth [m]
EARTH_MU = 398600.4418E+09 # gravitational parameter of the Earth [m^3/s^2]
EARTH_J2 = 1.082626925639e-3 # J2 constant of the Earth [-]

AU = 1.495978707E+11 # astronomical unit [m]

# the simulation starts on 31 Aug, 2026 at 0:00
REF_YEAR, REF_MONTH, REF_DAY = 2026., 8., 31.
term1 = int(REF_YEAR/100)
term2 = 2 - term1 + int(term1/4)
REF_JD = int(365.25*(REF_YEAR+4716)) + int(30.6001*(REF_MONTH+1)) + REF_DAY + term2 - 1524.5
REF_T = (REF_JD - 2451545.)/36525