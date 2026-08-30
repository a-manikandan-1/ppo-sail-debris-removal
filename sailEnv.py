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
    getSolarPosition,
    meoe2coe,
    sailDynamics,
    wrapCircularAngle,
)

""" CLASSES """
class OrbitRaisingEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(self, dt: float=60, maxTime: float=365*86400., sailAltitude0: float=7E+05, debrisAltitude0: float=1E+06, inc0: float=np.deg2rad(60), raan0: float=0.0, ac: float=1E-04, coneRateMax: float=np.deg2rad(0.5), clockRateMax: float=np.deg2rad(0.5), altThreshold: float=500.0, divergenceFactor: int=100):
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
            altThreshold: tolerance to call successful termination [m]
            divergenceFactor: how many times larger than the initial altitude separation to call divergence [-]
        """

        super().__init__()
        
        # simulation parameters
        self.dt = dt
        self.maxTime = maxTime
        self.altThreshold = altThreshold
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

        # observation space: [dAlt, alphan, sindelta, cosdelta] in normalized units
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, -1.0, -1.0]),
            high = np.array([np.inf, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # placeholders
        self.sailAltitude = sailAltitude0
        self.sailState = np.zeros(6, dtype=float)
        self.coneAngle = 0.0
        self.clockAngle = 0.0

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

        self.sailAltitude = self.sailAltitude0
        self.sailState = coe2meoe(a=EARTH_RADIUS+self.sailAltitude0, e=0.0, i=self.inc0, raan=self.raan0, aop=0.0, nu=sailL-self.raan0)

        self.prevAltDiff = np.abs(self.sailAltitude - self.debrisAltitude0)
        self.altDiff = self.prevAltDiff

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
        self.sailAltitude = meoe2coe(*self.sailState)[0] - EARTH_RADIUS
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
        dAlt = self.sailAltitude - self.debrisAltitude0

        obs = np.array([
            dAlt,
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
            "sailAltitude": self.sailAltitude,
            "altDiff": self.altDiff,
            "coneAngle": self.coneAngle,
            "clockAngle": self.clockAngle
        }

        return info

    def _checkTerminationAndReward(self):
        """
            Checks if the solar sail has reached the debris altitude.

            Calculates the reward for the current step following the expression:

            Reward =  1.0 * altitude difference normalized
                    - 0.01 * altitude difference in threshold units
                    + 100 * [if within threshold]
                    - 100 * [if states are diverging]
        """
        self.altDiff = np.abs(self.sailAltitude - self.debrisAltitude0)

        safeTerminate = (self.altDiff <= self.altThreshold)
        unsafeTerminate = (self.altDiff >= self.divergenceDistance)

        terminated = safeTerminate or unsafeTerminate

        altThresholdNorm = self.altDiff/self.distance0
        altChangeNorm = (self.prevAltDiff - self.altDiff)/self.distance0

        reward = 1.0*altChangeNorm - 0.01*altThresholdNorm + (100 if safeTerminate else 0) + (-100 if unsafeTerminate else 0)

        self.prevAltDiff = self.altDiff

        return bool(terminated), reward

""" REGISTRATION """
register(id="OrbitRaising-v1", entry_point="sailEnv:OrbitRaisingEnv")