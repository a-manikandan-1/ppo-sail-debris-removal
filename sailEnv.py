"""
    Create the custom Gymnasium environments to simulate solar sail dynamics
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from scipy.integrate import solve_ivp

from constants import *
from utils import (
    coe2meoe,
    debrisDynamics,
    getSolarPosition,
    meoe2coe,
    meoe2rv,
    sailDynamics,
    wrapCircularAngle,
)

""" CLASSES """
class OrbitRaisingEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(self, dt: float=60, maxTime: float=365*86400., sailAltitude0: float=7E+05, debrisAltitude0: float=1E+06, inc0: float=np.deg2rad(60), raan0: float=0.0, ac: float=1E-04, coneRateMax: float=np.deg2rad(0.5), clockRateMax: float=np.deg2rad(0.5), eScale: float=0.05, hScale: float=0.01, kScale: float=0.01, posThreshold: float=1000, velThreshold: float=1, divergenceFactor: int=100):
        """
            Environment simulating the raising of a solar sail from 700 km to 1000 km. Dynamics follow from the variational equations in MEOEs with the J2 effect and SRP as perturbations.

            PARAMETERS
            ----------
            dt: time step [s]
            maxTime: maximum propagation time [s]
            sailAltitude0: initial altitude of the solar sail [m]
            debrisAltitude0: initial altitude of the debris object [m]
            inc0: initial inclination [rad]
            raan0: initial RAAN [raad]
            ac: characteristic acceleration of the solar sail [m/s^2]
            coneRateMax: maximum slew rate of the cone angle [rad/s]
            clockRateMax: maximum slew rate of the clock angle [rad/s]
            eScale: normalization factor for f and g [-]
            hScale: normalization factor for h [-]
            kScale: normalization factor for k [-]
            posThreshold: relative distance reward threshold to indicate CW equations validity [m]
            velThreshold: relative velocity reward threshold to indicate CW equations validity [m/s]
            divergenceFactor: how many times larger than the initial altitude separation to call divergence [-]
        """

        super().__init__()
        
        # simulation parameters
        self.dt = dt
        self.maxTime = maxTime
        self.posThreshold = posThreshold
        self.velThreshold = velThreshold 
        self.divergenceFactor = divergenceFactor

        # problem parameters
        self.sailAltitude0 = sailAltitude0
        self.debrisAltitude0 = debrisAltitude0
        self.distance0 = np.abs(self.debrisAltitude0 - self.sailAltitude0)
        self.divergenceDistance = self.distance0 * self.divergenceFactor
        self.inc0 = inc0
        self.raan0 = raan0
        self.ac = ac
        self.coneRateMax = coneRateMax
        self.clockRateMax = clockRateMax

        # action space: [alphaDot, deltaDot] in normalized units
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) 

        # observation space: [dp, df, dg, dh, dk, sindL, cosdL, alphan, sindelta, cosdelta] in normalized units
        self.eScale = eScale
        self.hScale = hScale
        self.kScale = kScale
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1.0, -1.0, 0.0, -1.0, -1.0]),
            high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # placeholders
        self.sailState = np.zeros(6, dtype=float)
        self.debrisState = np.zeros(6, dtype=float)
        self.coneAngle = 0.0
        self.clockAngle = 0.0
        self.history = {}

    def reset(self, *, seed: int | None=None, options: dict | None=None):
        """
            Intializes a new episode.

            PARAMETERS
            ----------
            seed: used to initialize the RNG

            RETURNS
            ----------
            obs: observation vector
            info: additional information
        """
        super().reset(seed=seed)
        rng = self.np_random

        self.t = 0.0

        # randomize sail angular position and orientation 
        sailL = rng.uniform(0.0, 2*np.pi)
        self.coneAngle = rng.uniform(0.0, np.pi/2)
        self.clockAngle = rng.uniform(0.0, 2*np.pi)

        # randomize debris angular position
        debrisL = rng.uniform(0.0, 2*np.pi)

        self.sailState = coe2meoe(a=EARTH_RADIUS+self.sailAltitude0, e=0.0, i=self.inc0, raan=self.raan0, aop=0.0, nu=sailL-self.raan0)

        self.debrisState = coe2meoe(a=EARTH_RADIUS+self.debrisAltitude0, e=0.0, i=self.inc0, raan=self.raan0, aop=0.0, nu=debrisL-self.raan0)
        debrisSol = solve_ivp(debrisDynamics, (0.0, self.maxTime), self.debrisState, dense_output=True, rtol=1E-06, atol=1E-09)
        self._debrisSol = debrisSol.sol

        sailStateRV = meoe2rv(*self.sailState)
        debrisStateRV = meoe2rv(*self.debrisState)

        self.prevRelDist = np.linalg.norm(sailStateRV[:3] - debrisStateRV[:3])
        self.prevRelVel = np.linalg.norm(sailStateRV[3:] - debrisStateRV[3:])
        self.relDist = self.prevRelDist
        self.relVel = self.prevRelVel

        return self._getObs(), self._getInfo()

    def step(self, action):
        """
            Takes one step in the episode.

            PARAMETERS
            ----------
            action: action vector

            RETURNS
            ----------
            obs: observation vector
            reward: reward function value
            terminated: whether the states diverged
            truncated: whether the episode reached its maximum
            info: information
        """
        
        # convert action into control angles
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        coneRate = action[0]*self.coneRateMax
        clockRate = action[1]*self.clockRateMax

        # propagate dynamics
        y0Sail = self.sailState.copy()
        tspan = (self.t, self.t+self.dt)
        raSun, decSun, rSun = getSolarPosition(self.t)

        sailSol = solve_ivp(sailDynamics, tspan, y0Sail, args=(self.t, self.coneAngle, coneRate, self.clockAngle, clockRate, self.ac, raSun, decSun, rSun), rtol=1E-08, atol=1E-12)

        self.sailState = sailSol.y[:, -1]
        self.debrisState = self._debrisSol(self.t)
        self.t += self.dt
        self.coneAngle = np.clip(self.coneAngle + coneRate * self.dt, 0.0, np.pi/2)
        self.clockAngle = wrapCircularAngle(self.clockAngle + clockRate * self.dt)

        # get termination check and reward interpretation
        terminated, reward = self._checkTerminationAndReward()

        # check for step limit
        truncated = self.t >= self.maxTime

        return self._getObs(), reward, terminated, truncated, self._getInfo()

    def _getObs(self) -> np.ndarray:
        """
            Returns the observation vector.
        """
        dp, df, dg, dh, dk, dL = self.sailState-self.debrisState
        dL = ((dL+np.pi) % (2*np.pi)) - np.pi
        obs = np.array([
            dp/(self.sailAltitude0 - self.debrisAltitude0), 
            df/self.eScale, 
            dg/self.eScale, 
            dh/self.hScale, 
            dk/self.kScale, 
            np.sin(dL), 
            np.cos(dL), 
            self.coneAngle/(np.pi/2), 
            np.sin(self.clockAngle), 
            np.cos(self.clockAngle)
        ])

        return obs.astype(np.float32)

    def _getInfo(self) -> dict:
        """
            Returns information about the environment.
        """
        info = {
            "time": self.t,
            "sailState": self.sailState,
            "sailAltitude": meoe2coe(*self.sailState)[0] - EARTH_RADIUS,
            "debrisState": self.debrisState,
            "debrisAltitude": meoe2coe(*self.debrisState)[0] - EARTH_RADIUS,
            "relDist": self.relDist,
            "coneAngle": self.coneAngle,
            "clockAngle": self.clockAngle
        }

        return info

    def _checkTerminationAndReward(self):
        """
            Checks if the solar sail has reached a threshold region where the variational equations in MEOEs can be swapped for the Clohessy-Wiltshire equations.

            Calculates the reward for the current step following the expression:

            Reward =  1.0 * relative position change normalized
                    + 1.0 * relative velocity change normalized
                    - 0.01 * relative position in threshold units
                    - 0.01 * relative velocity in threshold units
                    - 0.01
                    + 100 * [if within threshold]
                    - 100 * [if states are diverging]
        """

        rvSail = meoe2rv(*self.sailState)
        rvDebris = meoe2rv(*self.debrisState)

        self.relDist = np.linalg.norm(rvSail[:3] - rvDebris[:3]) 
        self.relVel = np.linalg.norm(rvSail[3:] - rvDebris[3:])

        safeTerminate = (self.relDist <= self.posThreshold) and (self.relVel <= self.velThreshold)
        unsafeTerminate = (self.relDist >= self.divergenceDistance)

        terminated = safeTerminate or unsafeTerminate

        posThresholdNorm = self.relDist/self.posThreshold
        posChangeNorm = (self.prevRelDist - self.relDist)/self.posThreshold
        velThresholdNorm = self.relVel/self.velThreshold
        velChangeNorm = (self.prevRelVel - self.relVel)/self.velThreshold

        reward = 1.0*posChangeNorm + 1.0*velChangeNorm - 0.01*posThresholdNorm - 0.01*velThresholdNorm - 0.01 + (100 if safeTerminate else 0) + (-100 if unsafeTerminate else 0)

        self.prevRelDist = self.relDist
        self.prevRelVel = self.relVel

        return bool(terminated), reward

""" REGISTRATION """
register(id="OrbitRaising-v1", entry_point="sailEnv:OrbitRaisingEnv")