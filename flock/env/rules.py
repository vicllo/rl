"""Game rules: teams, catches, scores, done condition."""

import equinox as eqx
import jax
import jax.numpy as jnp

from flock.env.types import Agents, EnvConfig, TeamConfig
from flock.env.physics import pairwise_distances


class Rules(eqx.Module):
    """Base class for game rules. Defines teams and game mechanics."""
    teams: tuple[TeamConfig, ...]

    def interact(self, env_config: EnvConfig, teams: tuple[Agents, ...]) -> tuple[tuple[Agents, ...], tuple[jax.Array, ...]]:
        """Apply rules after physics. Returns (updated_teams, scores_per_team)."""
        raise NotImplementedError

    def is_done(self, teams: tuple[Agents, ...]) -> jax.Array:
        """Check if the episode should end (beyond max_steps)."""
        raise NotImplementedError


class PredatorPrey(Rules):
    """Standard predator-prey: team 0 (predators) catches team 1 (prey)."""
    catch_radius: float = 0.3

    def __init__(
        self,
        n_predators_max: int = 20,
        n_prey_max: int = 50,
        n_predators_init: int = 5,
        n_prey_init: int = 20,
        death_ratio_predator: float = 0.1,
        death_ratio_prey: float = 0,
        max_speed_predator: float = 1.0,
        max_speed_prey: float = 1.5,
        max_accel_predator: float = 8.0,
        max_accel_prey: float = 12.0,
        catch_radius: float = 0.3,
        k_teammates: int = 5,
        k_opponents: int = 5,
    ):
        self.teams = (
            TeamConfig("predators", n_predators_max, n_predators_init, death_ratio_predator, max_speed_predator, max_accel_predator, k_teammates, k_opponents),
            TeamConfig("prey", n_prey_max, n_prey_init, death_ratio_prey, max_speed_prey, max_accel_prey, k_teammates, k_opponents),
        )
        self.catch_radius = catch_radius

    def interact(self, env_config: EnvConfig, teams: tuple[Agents, ...]) -> tuple[tuple[Agents, ...], tuple[jax.Array, ...]]:
        predators, prey = teams

        dists = pairwise_distances(predators.pos, prey.pos, env_config.arena_size, a_mask=predators.alive, b_mask=prey.alive)
        in_range = dists < self.catch_radius
        catch_mask = in_range & prey.alive[None, :]
        caught = jnp.any(catch_mask, axis=0)
        new_prey_alive = prey.alive & ~caught

        # +1 for each predator within catch_radius of a caught prey (not split)
        pred_score = catch_mask.sum(axis=1).astype(jnp.float32)


        # Randomly kill predators with a small probability among the alive ones (simulate natural death)
        death_prob = env_config.dt * 5 
        random_vals = jax.random.uniform(jax.random.PRNGKey(0), shape=predators.alive.shape)
        natural_deaths = (random_vals < death_prob) & predators.alive
        new_predators_alive = predators.alive & ~natural_deaths
        new_predators = predators._replace(alive=new_predators_alive)
        

        # Reactivate dead predator slots, capped by the number of caught prey.
        nb_new_predators = jnp.minimum(
            caught.sum(dtype=jnp.int32),
            (~new_predators.alive).sum(dtype=jnp.int32),
        )
        dead_slots = ~new_predators.alive
        spawn_mask = dead_slots & (jnp.cumsum(dead_slots.astype(jnp.int32)) <= nb_new_predators)
        new_predators = new_predators._replace(alive=new_predators.alive | spawn_mask)

        new_prey = prey._replace(alive=new_prey_alive)
        return (new_predators, new_prey), (pred_score, jnp.zeros(prey.pos.shape[0]))
    
    def is_done(self, teams: tuple[Agents, ...]) -> jax.Array:
        """Done when all prey are dead."""
        prey = teams[1]
        return ~prey.alive.any()
