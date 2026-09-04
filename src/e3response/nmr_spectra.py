"""Synthesise NMR spectra from magnetic shielding tensors.

The chain this module implements is::

    σ_k (3×3 per atom)  →  resonance frequencies  →  lineshapes  →  spectrum(ppm)

Everything is written in pure JAX and is differentiable end-to-end with respect to
the tensors, hence — when the tensors come from a model that recomputes edge vectors
from ``nodes.positions`` — with respect to the atomic positions too. That is what makes
structure resolution against an *experimental* spectrum possible::

    loss(positions) = ‖ spectrum(model(positions)) − spectrum_exp ‖

which is the only observable available in a real deployment, where neither the
per-atom tensors nor the true geometry are known.

Two lineshape regimes are provided:

- :func:`mas_spectrum` — fast magic-angle spinning. Only the isotropic shielding
  ``σ_iso = tr(σ)/3`` survives, so each site contributes one narrow line. This is the
  right model for ²⁹Si MAS and, to first order, for ²³Na MAS.
- :func:`powder_spectrum` — static powder. The observed shielding for a crystallite
  whose B₀ direction is ``n`` in the molecular frame is ``σ(n) = nᵀ σ n``; the pattern
  is the average over orientations. Computed directly from the tensor, so it needs no
  Haeberlen convention and no sign choices, and it is exactly rotation-covariant.

**Referencing.** Experiments report chemical *shifts* δ, calculations produce
*shieldings* σ, related by ``δ = slope·σ + intercept``. The rigorous form is
``δ = (σ_ref − σ)/(1 − σ_ref·10⁻⁶)``, i.e. ``slope ≈ −1``, but in practice `slope` and
`intercept` are obtained by regressing computed σ against measured δ for a set of
reference compounds *at the same level of theory*. No default calibration is baked in
here — that would be a fabricated number, specific to functional, basis and code.
:func:`shielding_to_shift` defaults to ``δ = −σ``, which merely mirrors the axis.

**Quadrupolar nuclei.** ²³Na is spin 3/2, so a real ²³Na lineshape carries a
second-order quadrupolar contribution governed by the electric field gradient. The
EFG is *not* predicted by a shielding-tensor model, so it cannot be synthesised here:
these functions describe the chemical-shift contribution only. For ²³Na MAS this shows
up as sites appearing at their true δ_iso rather than at the quadrupole-shifted centre
of gravity, and as unmodelled asymmetric broadening. `extra_shift` and per-site `fwhm`
are provided as empirical hooks; a proper treatment needs an EFG model.
"""

import functools
from typing import Callable, Literal, Optional, TypeAlias, Union

import jax
import jax.numpy as jnp
import jraph
from tensorial.gcnn.keys import predicted

from . import keys

__all__ = (
    "isotropic_shielding",
    "shielding_to_shift",
    "gaussian",
    "lorentzian",
    "pseudo_voigt",
    "make_grid",
    "mas_spectrum",
    "powder_spectrum",
    "spectrum_from_graph",
    "normalise",
    "resample",
    "spectrum_loss",
)

Array: TypeAlias = jax.Array
LineshapeName: TypeAlias = Literal["gaussian", "lorentzian", "pseudo_voigt"]
NormaliseMode: TypeAlias = Literal["area", "max", "none"]

# fwhm = 2·sqrt(2·ln2)·σ for a Gaussian
_FWHM_TO_SIGMA: float = 1.0 / 2.3548200450309493


# ── Tensor → frequency ────────────────────────────────────────────────────────


def isotropic_shielding(tensors: Array) -> Array:
    """Isotropic shielding ``σ_iso = tr(σ)/3`` of each tensor.

    :param tensors: ``(..., 3, 3)`` shielding tensors, in ppm.
    :return: ``(...)`` isotropic values, in ppm.
    """
    return jnp.trace(tensors, axis1=-2, axis2=-1) / 3.0


def shielding_to_shift(sigma: Array, slope: float = -1.0, intercept: float = 0.0) -> Array:
    """Convert shieldings to chemical shifts through a linear referencing relation.

    ``δ = slope·σ + intercept``. See the module docstring: `slope` and `intercept`
    should come from a regression against measured shifts at the same level of theory.
    The default simply flips the axis (``δ = −σ``), which is enough whenever the
    comparison target is itself expressed in shielding.
    """
    return slope * sigma + intercept


def principal_components(tensors: Array) -> Array:
    """Eigenvalues of the symmetric part of each tensor, ascending.

    Only the symmetric part is observable in an NMR lineshape, so the antisymmetric
    component is discarded here as it is in :func:`powder_spectrum`.

    :param tensors: ``(..., 3, 3)``.
    :return: ``(..., 3)`` eigenvalues, ascending.
    """
    sym = 0.5 * (tensors + jnp.swapaxes(tensors, -1, -2))
    return jnp.linalg.eigvalsh(sym)


# ── Lineshapes (each normalised to unit area) ─────────────────────────────────


def gaussian(x: Array, centre: Array, fwhm: Array) -> Array:
    """Unit-area Gaussian of full width at half maximum `fwhm`."""
    sigma = fwhm * _FWHM_TO_SIGMA
    z = (x - centre) / sigma
    return jnp.exp(-0.5 * z * z) / (sigma * jnp.sqrt(2.0 * jnp.pi))


def lorentzian(x: Array, centre: Array, fwhm: Array) -> Array:
    """Unit-area Lorentzian of full width at half maximum `fwhm`."""
    half = 0.5 * fwhm
    return half / (jnp.pi * ((x - centre) ** 2 + half * half))


def pseudo_voigt(x: Array, centre: Array, fwhm: Array, eta: float = 0.5) -> Array:
    """Unit-area pseudo-Voigt: ``eta`` Lorentzian + ``1 − eta`` Gaussian.

    `eta` = 0 is a pure Gaussian (inhomogeneous / disorder broadening), `eta` = 1 a
    pure Lorentzian (homogeneous / T₂ broadening). Real MAS lines usually sit between.
    """
    return eta * lorentzian(x, centre, fwhm) + (1.0 - eta) * gaussian(x, centre, fwhm)


def _get_lineshape(
    lineshape: Union[LineshapeName, Callable], eta: float = 0.5
) -> Callable[[Array, Array, Array], Array]:
    if callable(lineshape):
        return lineshape
    if lineshape == "gaussian":
        return gaussian
    if lineshape == "lorentzian":
        return lorentzian
    if lineshape == "pseudo_voigt":
        return functools.partial(pseudo_voigt, eta=eta)
    raise ValueError(
        f"Unknown lineshape {lineshape!r}; expected 'gaussian', 'lorentzian', "
        "'pseudo_voigt' or a callable (x, centre, fwhm) -> intensity"
    )


# ── Orientations for powder averaging ─────────────────────────────────────────


def fibonacci_hemisphere(n: int) -> Array:
    """`n` approximately uniform directions on the upper hemisphere.

    A golden-spiral (Fibonacci) grid: ``z`` is sampled uniformly, which is exactly the
    uniform measure on the sphere, and the azimuth advances by the golden angle. Only a
    hemisphere is needed because ``σ(n) = σ(−n)``.

    The grid is a constant — no gradient flows through it — so the powder average stays
    differentiable with respect to the tensors alone.
    """
    i = jnp.arange(n, dtype=jnp.float32) + 0.5
    z = i / n
    phi = jnp.pi * (1.0 + jnp.sqrt(5.0)) * i
    r = jnp.sqrt(jnp.clip(1.0 - z * z, 0.0, 1.0))
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)


# ── Grids and spectra ─────────────────────────────────────────────────────────


def make_grid(lo: float, hi: float, n: int = 2048) -> Array:
    """Evenly spaced ppm axis with `n` points.

    NMR axes are conventionally plotted decreasing left to right; that is a plotting
    choice. Keep the grid ascending here and reverse it at plot time.
    """
    return jnp.linspace(lo, hi, n)


def _broadcast_per_site(value: Array, n_sites: int, name: str) -> Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim == 0:
        return jnp.full((n_sites,), value)
    if value.shape != (n_sites,):
        raise ValueError(f"{name} must be a scalar or have shape ({n_sites},), got {value.shape}")
    return value


def mas_spectrum(
    tensors: Array,
    grid: Array,
    fwhm: Union[float, Array] = 1.0,
    weights: Optional[Array] = None,
    *,
    lineshape: Union[LineshapeName, Callable] = "gaussian",
    eta: float = 0.5,
    slope: float = -1.0,
    intercept: float = 0.0,
    extra_shift: Optional[Array] = None,
) -> Array:
    """Fast-MAS spectrum: one line per site at its isotropic shift.

    :param tensors: ``(n_sites, 3, 3)`` shielding tensors, in ppm.
    :param grid: ``(n_points,)`` ppm axis, in the same units as the output of the
        referencing relation (i.e. shifts if `slope`/`intercept` convert to shifts).
    :param fwhm: linewidth, scalar or one value per site.
    :param weights: per-site amplitudes, scalar-broadcast or ``(n_sites,)``. Use these
        to drop padding nodes (0) or to encode site multiplicity / occupancy. Default
        is 1 for every site.
    :param lineshape: ``"gaussian"``, ``"lorentzian"``, ``"pseudo_voigt"`` or a
        callable ``(x, centre, fwhm) -> intensity``.
    :param eta: Lorentzian fraction, used only by ``"pseudo_voigt"``.
    :param slope, intercept: referencing, see :func:`shielding_to_shift`.
    :param extra_shift: optional ``(n_sites,)`` additive term applied *after*
        referencing — an empirical hook for effects this module does not model, such
        as the second-order quadrupolar shift of a spin > 1/2 nucleus.
    :return: ``(n_points,)`` intensities.
    """
    tensors = jnp.asarray(tensors)
    n_sites = tensors.shape[0]
    centres = shielding_to_shift(isotropic_shielding(tensors), slope, intercept)
    if extra_shift is not None:
        centres = centres + jnp.asarray(extra_shift)

    widths = _broadcast_per_site(fwhm, n_sites, "fwhm")
    amps = (
        jnp.ones((n_sites,))
        if weights is None
        else _broadcast_per_site(weights, n_sites, "weights")
    )

    shape_fn = _get_lineshape(lineshape, eta)
    lines = shape_fn(grid[None, :], centres[:, None], widths[:, None])
    return jnp.sum(amps[:, None] * lines, axis=0)


def powder_spectrum(
    tensors: Array,
    grid: Array,
    fwhm: Union[float, Array] = 1.0,
    weights: Optional[Array] = None,
    *,
    n_orientations: int = 1024,
    lineshape: Union[LineshapeName, Callable] = "gaussian",
    eta: float = 0.5,
    slope: float = -1.0,
    intercept: float = 0.0,
) -> Array:
    """Static powder pattern, averaged over crystallite orientations.

    For a crystallite whose B₀ lies along the unit vector ``n`` (molecular frame), the
    observed shielding is the quadratic form ``σ(n) = nᵀ σ n``. Averaging a broadened
    line over a uniform set of ``n`` reproduces the full chemical-shift-anisotropy
    powder pattern, including the ``η ≠ 0`` asymmetry, with no convention choices.

    Convergence is controlled by `n_orientations` against `fwhm`: too few orientations
    with a narrow line gives a visibly spiky pattern rather than a smooth one. A few
    hundred is enough for `fwhm` comparable to the span; use a few thousand for sharp
    lines over a wide anisotropy.

    Memory stays bounded by scanning over sites, so the cost is
    ``O(n_orientations · n_points)`` at a time regardless of how many atoms there are.

    Parameters are as in :func:`mas_spectrum`.

    :return: ``(n_points,)`` intensities.
    """
    tensors = jnp.asarray(tensors)
    n_sites = tensors.shape[0]
    widths = _broadcast_per_site(fwhm, n_sites, "fwhm")
    amps = (
        jnp.ones((n_sites,))
        if weights is None
        else _broadcast_per_site(weights, n_sites, "weights")
    )

    directions = fibonacci_hemisphere(n_orientations)          # (n_orient, 3)
    shape_fn = _get_lineshape(lineshape, eta)
    norm = 1.0 / n_orientations

    def _site(carry, site):
        tensor, width, amp = site
        sym = 0.5 * (tensor + tensor.T)
        # σ(n) = nᵀ σ n for every orientation at once
        freqs = jnp.einsum("oi,ij,oj->o", directions, sym, directions)
        freqs = shielding_to_shift(freqs, slope, intercept)
        lines = shape_fn(grid[None, :], freqs[:, None], width)
        return carry + amp * norm * jnp.sum(lines, axis=0), None

    total, _ = jax.lax.scan(
        _site, jnp.zeros_like(grid, dtype=jnp.float32), (tensors, widths, amps)
    )
    return total


# ── Graph-level convenience ───────────────────────────────────────────────────


def graph_node_weights(graph: jraph.GraphsTuple) -> Optional[Array]:
    """Per-node weights that exclude padding nodes, or ``None`` if the graph is unpadded.

    A padded batch carries ``nodes.mask`` (``gcnn.keys.MASK``) marking real nodes;
    feeding it as `weights` keeps padding out of the spectrum.
    """
    mask = graph.nodes.get(keys.MASK)
    return None if mask is None else jnp.asarray(mask, dtype=jnp.float32)


def species_weights(
    graph: jraph.GraphsTuple, atomic_number: Optional[int] = None
) -> Optional[Array]:
    """Weights selecting one element, combined with the padding mask.

    An experimental spectrum is nucleus-specific (²³Na, ²⁹Si, …), so only the atoms of
    that element may contribute. ``None`` for `atomic_number` keeps every atom.
    """
    weights = graph_node_weights(graph)
    if atomic_number is None:
        return weights
    numbers = jnp.asarray(graph.nodes[keys.ATOMIC_NUMBERS]).reshape(-1)
    selected = (numbers == atomic_number).astype(jnp.float32)
    return selected if weights is None else selected * weights


def spectrum_from_graph(
    graph: jraph.GraphsTuple,
    grid: Array,
    *,
    atomic_number: Optional[int] = None,
    field: Optional[str] = None,
    mode: Literal["mas", "powder"] = "mas",
    **kwargs,
) -> Array:
    """Spectrum of one element's tensors taken straight from a graph.

    :param graph: graph carrying shielding tensors on its nodes.
    :param grid: ppm axis.
    :param atomic_number: restrict to this element (e.g. 11 for ²³Na, 14 for ²⁹Si).
        ``None`` uses every atom.
    :param field: node field holding the tensors. Defaults to the model's prediction
        (``predicted(nmr_tensors)``) and falls back to the labels (``nmr_tensors``).
    :param mode: ``"mas"`` → :func:`mas_spectrum`, ``"powder"`` → :func:`powder_spectrum`.
    :param kwargs: forwarded to the chosen spectrum function.
    """
    if field is None:
        field = predicted(keys.NMR_TENSORS)
        if field not in graph.nodes:
            field = keys.NMR_TENSORS
    if field not in graph.nodes:
        raise KeyError(
            f"Graph has no node field {field!r}; available: {sorted(graph.nodes)}"
        )

    tensors = jnp.asarray(graph.nodes[field])
    weights = species_weights(graph, atomic_number)
    fn = mas_spectrum if mode == "mas" else powder_spectrum
    return fn(tensors, grid, weights=weights, **kwargs)


# ── Comparison with an experimental spectrum ──────────────────────────────────


def normalise(y: Array, mode: NormaliseMode = "area", x: Optional[Array] = None) -> Array:
    """Normalise a spectrum so predicted and measured intensities are comparable.

    ``"area"`` divides by the trapezoidal integral (requires `x`, or assumes unit
    spacing), ``"max"`` divides by the peak height, ``"none"`` is the identity. Since
    experimental intensities are in arbitrary units, one of the first two is
    essentially always required before taking a difference.
    """
    if mode == "none":
        return y
    if mode == "max":
        return y / (jnp.max(jnp.abs(y)) + 1e-30)
    if mode == "area":
        area = jnp.trapezoid(y, x) if x is not None else jnp.sum(y)
        return y / (area + 1e-30)
    raise ValueError(f"Unknown normalise mode {mode!r}")


def resample(x_src: Array, y_src: Array, x_dst: Array) -> Array:
    """Linearly interpolate a measured spectrum onto the working grid.

    Experimental files come on their own axis; both spectra must share a grid before
    they can be subtracted. Points outside the source range are set to zero.
    """
    return jnp.interp(x_dst, x_src, y_src, left=0.0, right=0.0)


def wasserstein_distance(
    predicted_y: Array, target_y: Array, x: Optional[Array] = None
) -> Array:
    """1-Wasserstein (earth-mover) distance between two spectra, in ppm.

    Both spectra are treated as distributions of intensity over the ppm axis and
    compared through their cumulative integrals: ``W₁ = ∫ |CDF_p − CDF_t| dx``. Each
    CDF is rescaled to end at 1, so the result is independent of overall intensity.

    **Use this, not a point-by-point metric, when peaks may not overlap.** All the
    metrics in :func:`spectrum_loss` compare intensities grid point by grid point,
    which means that once a predicted line has moved more than roughly its own width
    away from the target line, the two no longer share any support: the difference
    stops changing and the gradient goes to zero. The loss is then flat and an
    optimiser is stuck, however far apart the peaks are. ``W₁`` instead measures how
    far intensity has to be *transported*, so it keeps growing — and keeps a non-zero
    gradient — at any separation, giving a basin of attraction as wide as the axis.

    The grid must be uniformly spaced (as produced by :func:`make_grid`).
    """
    dx = 1.0 if x is None else (x[1] - x[0])
    cp = jnp.cumsum(predicted_y)
    ct = jnp.cumsum(target_y)
    cp = cp / (cp[-1] + 1e-30)
    ct = ct / (ct[-1] + 1e-30)
    return jnp.sum(jnp.abs(cp - ct)) * jnp.abs(dx)


def spectrum_loss(
    predicted_y: Array,
    target_y: Array,
    *,
    x: Optional[Array] = None,
    normalise_mode: NormaliseMode = "area",
    metric: Literal["mse", "rmse", "cosine", "wasserstein"] = "rmse",
    fit_scale: bool = False,
) -> Array:
    """Scalar mismatch between a predicted and a measured spectrum.

    :param x: ppm axis, used by area normalisation and by ``"wasserstein"``.
    :param normalise_mode: applied to both spectra first — experimental intensities
        are arbitrary units, so comparing raw amplitudes is meaningless.
    :param metric:

        - ``"rmse"`` / ``"mse"`` — point-by-point difference.
        - ``"cosine"`` — ``1 − cos(pred, target)``, ignores overall scale entirely and
          responds to peak *positions* rather than heights.
        - ``"wasserstein"`` — transport distance in ppm, see
          :func:`wasserstein_distance`. The only option here whose gradient survives
          when the peaks do not overlap, hence the one to use for structure
          resolution unless the lines are broad enough to overlap from the start.

    :param fit_scale: if True, first rescale the prediction by the closed-form optimal
        factor ``⟨p,t⟩/⟨p,p⟩``. Differentiable, and it removes any residual amplitude
        mismatch that normalisation left behind. Irrelevant for ``"wasserstein"``,
        which normalises its own cumulatives.
    :return: scalar loss.

    Note that this compares spectra on a *shared grid*: a rigid ppm offset between
    calculation and experiment (a referencing error) shows up as a large loss even when
    the pattern is right. Fix the referencing via `slope`/`intercept` rather than
    expecting the optimiser to absorb it by moving atoms.
    """
    p = normalise(predicted_y, normalise_mode, x)
    t = normalise(target_y, normalise_mode, x)

    if metric == "wasserstein":
        return wasserstein_distance(p, t, x)

    if fit_scale:
        scale = jnp.sum(p * t) / (jnp.sum(p * p) + 1e-30)
        p = scale * p

    if metric == "cosine":
        num = jnp.sum(p * t)
        den = jnp.linalg.norm(p) * jnp.linalg.norm(t) + 1e-30
        return 1.0 - num / den

    sq = jnp.mean((p - t) ** 2)
    if metric == "mse":
        return sq
    if metric == "rmse":
        return jnp.sqrt(sq)
    raise ValueError(f"Unknown metric {metric!r}")
