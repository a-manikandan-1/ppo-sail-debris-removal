"""
    Utility helper functions
"""

import numpy as np

from constants import *


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

def srpRTN(y: np.ndarray, ac: float, coneAngle: float, clockAngle: float) -> np.ndarray:
    """
        SRP perturbing acceleration following an ideal model in the RTN frame.

        PARAMETERS
        ----------
        y: state vector in MEOEs
        ac: characteristic acceleration [m/s^2]
        coneAngle: cone angle [rad]
        clockAngle: clock angle [rad]

        RETURNS
        ----------
        srpVec: SRP perturbation
    """
    srpVec = ac*(np.cos(coneAngle)**2)*np.array([
        np.cos(coneAngle),
        np.sin(coneAngle)*np.sin(clockAngle),
        np.sin(coneAngle)*np.cos(clockAngle)
    ])

    return srpVec

def sailDynamics(t: float, y: np.ndarray, t0: float, coneAngle0: float, coneRate: float, clockAngle0: float, clockRate: float, ac: float) -> np.ndarray:
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

        RETURNS
        ----------
        meoeDot: derivatives of the MEOEs
    """
    dt = t - t0
    coneAngle = np.clip(coneAngle0 + coneRate * dt, 0.0, np.pi/2)
    clockAngle = np.clip(clockAngle0 + clockRate * dt, 0.0, 2*np.pi)
    ap = j2RTN(y) + srpRTN(y, ac, coneAngle, clockAngle)

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