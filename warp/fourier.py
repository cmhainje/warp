import jax
import jax.numpy as jnp


def wavenumber(j, L):
    return ((j + 1) // 2) * jnp.pi / L


def basis_1d(j, x, L):
    wn = wavenumber(j, L)
    phase = (-jnp.pi / 2) * (j % 2)
    return jnp.cos(wn * x + phase)


def basis_2d(jx, jy, x, y, Lx, Ly):
    return basis_1d(jx, x, Lx) * basis_1d(jy, y, Ly)


basis_2d = jax.vmap(basis_2d, in_axes=(0, 0, None, None, None, None), out_axes=0)


def basis_indices(orders):
    j_maxs = [2 * order + 1 for order in orders]
    idx = [
        (jx, jy)
        for jx in range(j_maxs[0])
        for jy in range(j_maxs[1])
    ]
    return jnp.array(idx)
