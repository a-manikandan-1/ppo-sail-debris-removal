"""
    Utility helper functions
"""

import numpy as np

from constants import *


def crossProduct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
        Computes the cross product of the given vectors. Explicitly defined to reduce computational time.

        PARAMETERS
        ----------
        a: vector 1
        b: vector 2

        RETURNS
        ----------
        c: cross product
    """
    c = np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

    return c

def wrapCircularAngle(angle: float, deg: bool=False) -> float:
    """
        Appropriately wraps angles that are bounded between 0 and 2*pi.

        PARAMETERS
        ----------
        angle: input angle [rad]

        RETURNS
        ----------
        angle: wrapped angle [rad]
    """
    if deg:
        angle = angle % 360.
        if angle < 0:
            angle += 360.
    else:
        angle = angle % (2*np.pi)
        if angle < 0:
            angle += 2*np.pi

    return angle


def coe2meoe(a: float, e: float, i: float, raan: float, aop: float, nu: float) -> np.ndarray:
    """
        Converts the classical orbital elements to the modified equinoctial orbit elements.

        PARAMETERS
        ----------
        a: semi-major axis [m]
        e: eccentricity [-]
        i: inclination [rad]
        raan: right ascension of the ascending node [rad]
        aop: argument of periapsis [rad]
        nu: true anomaly [rad]

        RETURNS
        ----------
        meoeVec = [p, f, g, h, k, L]
    """
    p = a*(1 - e**2)
    f = e*np.cos(raan+aop)
    g = e*np.sin(raan+aop)
    h = np.tan(i/2.)*np.cos(raan)
    k = np.tan(i/2)*np.sin(raan)
    L = (raan+aop+nu)%(2*np.pi)

    meoeVec = np.array([p, f, g, h, k, L])

    return meoeVec

def meoe2coe(p: float, f: float, g: float, h: float, k: float, L: float) -> np.ndarray:
    """
        Converts the modified equinoctial orbit elements to the classical orbital elements.

        PARAMETERS
        ----------
        p: first MEOE [m]
        f: second MEOE [-]
        g: third MEOE [-]
        h: fourth MEOE [-]
        k: fifth MEOE [-]
        L: sixth MEOE [rad]

        RETURNS
        ----------
        coeVec = [a, e, i, raan, aop, nu]
    """
    e = np.sqrt(f**2 + g**2)
    a = p/(1-e**2)
    i = 2*np.arctan(np.sqrt(h**2 + k**2))
    raan = np.arctan2(k, h)
    longitude = np.arctan2(g, f)
    aop = wrapCircularAngle(longitude-raan)
    nu = wrapCircularAngle(L-longitude)
    raan = wrapCircularAngle(raan)

    return np.array([a, e, i, raan, aop, nu])

def meoe2rv(p: float, f: float, g: float, h: float, k: float, L: float) -> np.ndarray:
    """
        Converts the modified equinoctial orbit elements to the radius and velocity vectors.

        PARAMETERS
        ----------
        p: first MEOE [m]
        f: second MEOE [-]
        g: third MEOE [-]
        h: fourth MEOE [-]
        k: fifth MEOE [-]
        L: sixth MEOE [rad]

        RETURNS
        ----------
        rvVec = [rVec vVec]
    """
    q = 1 + f*np.cos(L) + g*np.sin(L)
    s2 = 1 + h**2 + k**2
    r = p/q
    
    term1 = np.sqrt(EARTH_MU/p)
    vec1 = np.array([1-k**2+h**2, 2*k*h, -2*k]) / s2
    vec2 = np.array([2*k*h, 1+k**2-h**2, 2*h]) / s2
    x = r*np.cos(L)
    y = r*np.sin(L)
    xDot = -term1 * (g + np.sin(L))
    yDot = term1 * (f + np.cos(L))

    rVec = x*vec1 + y*vec2
    vVec = xDot*vec1 + yDot*vec2

    return np.concatenate((rVec, vVec))

def meoeVariationalEqns(y: np.ndarray, ap: np.ndarray) -> np.ndarray:
    """
        Returns the variational equations in MEOEs

        PARAMETERS
        ----------
        y: state vector in MEOEs
        ap: perturbing acceleration in the RTN frame

        RETURNS
        ----------
        meoeDot: meoe derivatives
    """
    p, f, g, h, k, L = y
    aR, aT, aN = ap

    q = 1 + f*np.cos(L) + g*np.sin(L)
    s2 = 1 + h**2 + k**2
    term1 = np.sqrt(p/EARTH_MU)

    meoeDot = term1*np.array([
        (2.0*p/q)*aT,
        np.sin(L)*aR + ((1+q)*np.cos(L) + f)*aT/q - (g*(h*np.sin(L) - k*np.cos(L)))*aN/q,
        -np.cos(L)*aR + ((1+q)*np.sin(L) + g)*aT/q + (f*(h*np.sin(L) - k*np.cos(L)))*aN/q,
        (s2*np.cos(L))*aN/(2*q),
        (s2*np.sin(L))*aN/(2*q),
        (h*np.sin(L) - k*np.cos(L))*aN/q + (1/term1)*np.sqrt(EARTH_MU*p)*((q/p)**2)
    ])

    return meoeDot

def j2RTN(y: np.ndarray) -> np.ndarray:
    """
        J2 perturbing acceleration in the RTN frame.

        PARAMETERS
        ----------
        y: state vector in MEOEs

        RETURNS
        ----------
        j2Vec: J2 perturbation
    """
    p, f, g, h, k, L = y

    q = 1 + f*np.cos(L) + g*np.sin(L)
    s2 = 1 + h**2 + k**2
    r = p/q
    term1 = -(3*EARTH_MU*EARTH_J2*(EARTH_RADIUS**2))/(2*(r**4)*(s2**2))
    term2 = h*np.sin(L) - k*np.cos(L)

    j2Vec = term1*np.array([
        s2**2 - 12*term2**2,
        8*term2*(h*np.cos(L) + k*np.sin(L)),
        4*(1-h**2-k**2)*term2
    ])

    return j2Vec

def getSolarPosition(t: float) -> tuple:
    """
        Calculates the position of the Sun. Following the technique outlined at https://squarewidget.com/solar-coordinates/.

        PARAMETERS
        ----------
        t: seconds since reference epoch [s]

        RETURNS
        ----------
        ra: right ascension of the Sun [rad]
        dec: declination of the Sun [rad]
        r: Earth-Sun distance [m]
    """
    t = REF_T + t/(36525*86400)
    L0 = wrapCircularAngle(280.46646 + (36000.76983*t) + (0.0003032*(t**2)), deg=True) # geometric mean longitude of the Sun
    M = wrapCircularAngle(357.52911 + (35999.05029*t) - (0.0001537*(t**2)), deg=True) # mean anomaly of the Sun
    # eEarth = 0.016708634 - (0.000042037*t) - (0.0000001267*(t**2)) # eccentricity of the Earth's orbit
    sunCenter = (1.914602 - (0.004817*t) - (0.000014*(t**2))) * np.sin(np.deg2rad(M)) + (0.019993 - (0.000101*t))*np.sin(np.deg2rad(M)*2) + (0.000289*np.sin(np.deg2rad(M)*3))
    Ltrue = wrapCircularAngle(L0+sunCenter, deg=True) # true longitude of the Sun
    # nu = wrapCircularAngle(M+sunCenter, deg=True) # true anomaly of the Sun
    r = 1.00014 - 0.01671*np.cos(np.deg2rad(M)) - 0.00014*np.cos(2*np.deg2rad(M)) # radius between Sun and Earth in AU
    omega = 125.04 - (1934.136*t) # correction for nutation and aberration
    Lapp = Ltrue - 0.00569 - (0.00478*np.sin(np.deg2rad(omega))) # apparent longitude of the Sun
    U = t/100
    e0 = (23+26/60+21.448/3600) - (4680.93/3600)*U - 1.55*np.power(U, 2) + 1999.25*np.power(U, 3) - 51.38*np.power(U, 4) - 249.67*np.power(U, 5) - 39.05*np.power(U, 6) + 7.12*np.power(U, 7) + 27.87*np.power(U, 8) + 5.79*np.power(U, 9) + 2.45*np.power(U, 10) # obliquity of the Earth's orbit
    eCorrected = e0 + 0.00256*np.cos(np.deg2rad(omega)) # correction for parallax

    # right ascension of the Sun
    ra = np.arctan2(np.cos(np.deg2rad(eCorrected))*np.sin(np.deg2rad(Lapp)), np.cos(np.deg2rad(Lapp)))

    # declination of the Sun
    dec = np.arcsin(np.sin(np.deg2rad(eCorrected))*np.sin(np.deg2rad(Lapp)))

    return ra, dec, r*AU

def sailNormalRTN(y: np.ndarray, raSun: float, decSun: float, coneAngle: float, clockAngle: float) -> np.ndarray:
    """
        Returns the normal vector of the solar sail in the RTN frame.

        PARAMETERS
        ----------
        y: state vector in MEOEs
        raSun: right ascension of the Sun [rad]
        decSun: declination of the Sun [rad]
        coneAngle: cone angle [rad]
        clockAngle: clock angle [rad]

        RETURNS
        ----------
        nVecRTN: sail normal vector in the RTN frame
    """
    Rsg = np.array([
        [-np.cos(raSun)*np.cos(decSun), np.sin(raSun), -np.cos(raSun)*np.sin(decSun)],
        [-np.sin(raSun)*np.cos(decSun), -np.cos(raSun), -np.sin(raSun)*np.sin(decSun)],
        [-np.sin(decSun), 0, np.cos(decSun)]
    ])
    nVecS = np.array([np.cos(coneAngle), np.sin(coneAngle)*np.sin(clockAngle), np.sin(coneAngle)*np.cos(clockAngle)])
    nVecECI = Rsg @ nVecS
    
    rvVec = meoe2rv(*y)
    rVec, vVec = rvVec[:3], rvVec[3:]
    R = rVec/np.linalg.norm(rVec)
    N = crossProduct(rVec, vVec)
    N /= np.linalg.norm(N)
    T = crossProduct(N, R)
    Rgrtn = np.array([R, T, N])

    nVecRTN = Rgrtn @ nVecECI
    nVecRTN /= np.linalg.norm(nVecRTN)

    return nVecRTN

def isShadowed(y: np.ndarray, raSun: float, decSun: float, rSun: float) -> int:
    """
        Checks if the sail is shadowed through the cylindrical shadow model.
    
        PARAMETERS
        ----------
        y: state vector in MEOEs
        raSun: right ascension of the Sun [rad]
        decSun: declination of the Sun [rad]
        rSun: radial Earth-Sun distance [m]

        RETURNS
        ----------
        eta: is shadowed
    """
    sunVec = rSun*np.array([np.cos(decSun)*np.cos(raSun), np.cos(decSun)*np.sin(raSun), np.sin(decSun)])
    sunMag = np.linalg.norm(sunVec)
    sailVec = meoe2rv(*y)[:3]
    sailMag = np.linalg.norm(sailVec)

    theta = np.arccos(np.dot(sunVec, sailVec)/(sunMag*sailMag))
    thetaSun = np.arccos(EARTH_RADIUS/sunMag)
    thetaSail = np.arccos(EARTH_RADIUS/sailMag)

    if thetaSun + thetaSail < theta:
        eta = 0
    else:
        eta = 1

    return eta

def srpRTN(y: np.ndarray, ac: float, coneAngle: float, clockAngle: float, raSun: float, decSun: float, rSun: float) -> np.ndarray:
    """
        SRP perturbing acceleration following an ideal model in the RTN frame.

        PARAMETERS
        ----------
        y: state vector in MEOEs
        ac: characteristic acceleration [m/s^2]
        coneAngle: cone angle [rad]
        clockAngle: clock angle [rad]
        raSun: right ascension of the Sun [rad]
        decSun: declination of the Sun [rad]
        rSun: radial Earth-Sun distance [m]

        RETURNS
        ----------
        srpVec: SRP perturbation
    """
    nVecRTN = sailNormalRTN(y, raSun, decSun, coneAngle, clockAngle)
    shadow = isShadowed(y, raSun, decSun, rSun)
    srpVec = shadow*ac*(np.cos(coneAngle)**2)*nVecRTN

    return srpVec

def sailDynamics(t: float, y: np.ndarray, t0: float, coneAngle0: float, coneRate: float, clockAngle0: float, clockRate: float, ac: float, raSun: float, decSun: float, rSun: float) -> np.ndarray:
    """
        Solar sail dynamics function to be passed into solve_ivp

        PARAMETERS
        ----------
        t: time [s]
        y: state vector in MEOEs
        t0: initial time [s]
        coneAngle0: initial cone angle [rad]
        coneRate: slew rate of the cone angle [rad/s]
        clockAngle0: intiial clock angle [rad]
        clockRate: slew rate of the clock angle [rad/s]
        ac: characteristic acceleration [m/s^2]
        raSun: right ascension of the Sun [rad]
        decSun: declination of the Sun [rad]
        rSun: radial Earth-Sun distance [m]

        RETURNS
        ----------
        meoeDot: derivatives of the MEOEs
    """
    dt = t - t0
    coneAngle = np.clip(coneAngle0 + coneRate * dt, 0.0, np.pi/2)
    clockAngle = wrapCircularAngle(clockAngle0 + clockRate * dt)
    ap = j2RTN(y) + srpRTN(y, ac, coneAngle, clockAngle, raSun, decSun, rSun)

    return meoeVariationalEqns(y, ap)

def debrisDynamics(t: float, y: np.ndarray) -> np.ndarray:
    """
        Debris object dynamics function to be passed into solve_ivp

        PARAMETERS
        ----------
        t: time [s]
        y: state vector in MEOEs
        
        RETURNS
        ----------
        meoeDot: derivatives of the MEOEs
    """
    ap = j2RTN(y)

    return meoeVariationalEqns(y, ap)