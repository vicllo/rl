import jax.numpy as jnp


def clamp_magnitude(v, max_mag):
    """Clamp vector magnitude, preserving direction."""
    mag = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.where(mag > max_mag, v / mag * max_mag, v)


def integrate(pos, vel, accel, dt, max_speed):
    """Euler integration with velocity clamping."""
    vel = clamp_magnitude(vel + accel * dt, max_speed)
    pos = pos + vel * dt
    return pos, vel


def wrap_position(pos, arena_size):
    """Toroidal wrapping — agents exiting one side appear on the other."""
    return pos % arena_size


def pairwise_distances(pos_a, pos_b, arena_size, a_mask=None, b_mask=None):
    """Toroidal pairwise distances between two sets of agents.

    Returns distances of shape (n_a, n_b), accounting for wrap-around.
    """
    # (n_a, 1, 2) - (1, n_b, 2) -> (n_a, n_b, 2)
    delta = pos_a[:, None, :] - pos_b[None, :, :]
    # Shortest distance on torus
    delta = delta - arena_size * jnp.round(delta / arena_size)
    distances = jnp.linalg.norm(delta, axis=-1)

    # Apply masks if provided
    if a_mask is not None:
        distances = distances.at[:, :].set(jnp.where(a_mask[:, None], distances, 1e6))
    if b_mask is not None:
        distances = distances.at[:, :].set(jnp.where(b_mask[None, :], distances, 1e6))

    return distances


def pairwise_deltas(pos_a, pos_b, arena_size):
    """Toroidal pairwise displacement vectors from pos_b to pos_a.

    Returns (n_a, n_b, 2) — the direction from each b to each a,
    accounting for wrap-around.
    """
    delta = pos_a[:, None, :] - pos_b[None, :, :]
    return delta - arena_size * jnp.round(delta / arena_size)
