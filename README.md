# Using PPO to Optimize Solar Sail Trajectories for LEO Debris Rendezvous

Final assignment for AE4350 Bio-inspired Intelligence and Learning for Aerospace Applications.

Orbital debris has been a growing problem in the near-Earth sphere, with no effective means of cleaning their numbers. A way to implement active debris removal that is cost-effective and uses a novel approach is by using solar sails. This study analyzes the use of proximal policy optimization algorithms in controlling solar sail trajectories to rendezvous with a debris object.

A custom simulated environment is defined according to the [Gymnasium framework](https://gymnasium.farama.org/introduction/create_custom_env/) and the PPO algorithm is implemented through the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) library. This preliminary study assumes circular orbits, an ideal SRP model and the only other source of perturbation being the J2 effect.

## Requirements

The study is conducted entirely using Python and makes use of the following libraries.

```
    numpy==2.5.2
    matplotlib==3.11.1
    seaborn==0.13.2
    scipy==1.18.1
    gymnasium==1.2.2
    gymnasium[box2d]==1.2.2
    stable-baselines3==2.9.0
    stable-baselines3[extra]==2.9.0
    optuna==4.9.0
    notebook==7.6.2
```