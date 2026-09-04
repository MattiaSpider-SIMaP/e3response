import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from e3response import keys, nmr_spectra


def _tensor(iso: float, aniso: float = 0.0, asym: float = 0.0) -> np.ndarray:
    """Diagonal shielding tensor with the requested isotropic value and anisotropy."""
    return np.diag(
        [iso - 0.5 * aniso * (1 + asym), iso - 0.5 * aniso * (1 - asym), iso + aniso]
    ).astype(np.float32)


def _random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


# ── lineshapes ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape_fn", [nmr_spectra.gaussian, nmr_spectra.lorentzian])
def test_lineshapes_have_unit_area(shape_fn):
    """Unit area on the full line — compared against the analytically truncated value.

    A Gaussian is numerically complete well inside the window, but the Lorentzian's
    tails are heavy: over |x| <= X only (2/pi)*arctan(2X/fwhm) of its area is captured
    (0.9984 here), which is physics rather than a normalisation error.
    """
    half_width, fwhm = 400.0, 2.0
    x = jnp.linspace(-half_width, half_width, 200_001)
    y = shape_fn(x, 0.0, fwhm)

    if shape_fn is nmr_spectra.lorentzian:
        expected = float(2.0 / np.pi * np.arctan(2.0 * half_width / fwhm))
    else:
        expected = 1.0
    assert float(jnp.trapezoid(y, x)) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("shape_fn", [nmr_spectra.gaussian, nmr_spectra.lorentzian])
def test_lineshape_fwhm_is_the_full_width_at_half_maximum(shape_fn):
    fwhm = 3.0
    peak = float(shape_fn(jnp.array(0.0), 0.0, fwhm))
    half = float(shape_fn(jnp.array(0.5 * fwhm), 0.0, fwhm))
    assert half == pytest.approx(0.5 * peak, rel=1e-5)


def test_pseudo_voigt_interpolates_between_its_limits():
    x = jnp.linspace(-20.0, 20.0, 4001)
    g = nmr_spectra.gaussian(x, 1.0, 2.0)
    lo = nmr_spectra.lorentzian(x, 1.0, 2.0)
    np.testing.assert_allclose(nmr_spectra.pseudo_voigt(x, 1.0, 2.0, eta=0.0), g, rtol=1e-6)
    np.testing.assert_allclose(nmr_spectra.pseudo_voigt(x, 1.0, 2.0, eta=1.0), lo, rtol=1e-6)


def test_unknown_lineshape_raises():
    with pytest.raises(ValueError):
        nmr_spectra.mas_spectrum(
            jnp.zeros((1, 3, 3)), nmr_spectra.make_grid(-1.0, 1.0, 8), lineshape="voigt"
        )


# ── MAS ───────────────────────────────────────────────────────────────────────


def test_mas_peak_sits_at_the_referenced_isotropic_shift():
    tensors = jnp.asarray(_tensor(80.0)[None])
    grid = nmr_spectra.make_grid(-200.0, 200.0, 4001)
    y = nmr_spectra.mas_spectrum(tensors, grid, fwhm=2.0)
    # default referencing is delta = -sigma
    assert float(grid[int(jnp.argmax(y))]) == pytest.approx(-80.0, abs=0.2)


def test_mas_referencing_is_applied():
    tensors = jnp.asarray(_tensor(80.0)[None])
    grid = nmr_spectra.make_grid(-200.0, 200.0, 4001)
    y = nmr_spectra.mas_spectrum(tensors, grid, fwhm=2.0, slope=-1.0, intercept=100.0)
    assert float(grid[int(jnp.argmax(y))]) == pytest.approx(20.0, abs=0.2)


def test_mas_ignores_anisotropy_and_orientation():
    """Fast MAS keeps only the trace, so rotating a site cannot change the spectrum."""
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    base = _tensor(30.0, aniso=40.0, asym=0.4)
    rot = _random_rotation()
    rotated = rot @ base @ rot.T

    y_base = nmr_spectra.mas_spectrum(jnp.asarray(base[None]), grid, fwhm=3.0)
    y_rot = nmr_spectra.mas_spectrum(jnp.asarray(rotated[None]), grid, fwhm=3.0)
    np.testing.assert_allclose(np.asarray(y_base), np.asarray(y_rot), atol=1e-5)


def test_mas_sites_add_and_weights_select():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 4001)
    tensors = jnp.asarray(np.stack([_tensor(10.0), _tensor(60.0)]))

    both = nmr_spectra.mas_spectrum(tensors, grid, fwhm=2.0)
    first = nmr_spectra.mas_spectrum(tensors, grid, fwhm=2.0, weights=jnp.array([1.0, 0.0]))
    second = nmr_spectra.mas_spectrum(tensors, grid, fwhm=2.0, weights=jnp.array([0.0, 1.0]))

    np.testing.assert_allclose(np.asarray(both), np.asarray(first + second), atol=1e-6)
    # a zero-weighted site contributes nothing at its own position
    assert float(first[int(jnp.argmin(jnp.abs(grid + 60.0)))]) == pytest.approx(0.0, abs=1e-6)


def test_mas_area_equals_total_weight():
    grid = nmr_spectra.make_grid(-300.0, 300.0, 60_001)
    tensors = jnp.asarray(np.stack([_tensor(10.0), _tensor(-40.0)]))
    y = nmr_spectra.mas_spectrum(tensors, grid, fwhm=4.0, weights=jnp.array([1.0, 2.0]))
    assert float(jnp.trapezoid(y, grid)) == pytest.approx(3.0, rel=1e-3)


def test_extra_shift_moves_a_site():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 4001)
    tensors = jnp.asarray(_tensor(0.0)[None])
    y = nmr_spectra.mas_spectrum(tensors, grid, fwhm=2.0, extra_shift=jnp.array([-12.0]))
    assert float(grid[int(jnp.argmax(y))]) == pytest.approx(-12.0, abs=0.2)


def test_per_site_fwhm_broadens_only_that_site():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 8001)
    tensors = jnp.asarray(np.stack([_tensor(20.0), _tensor(-20.0)]))
    y = nmr_spectra.mas_spectrum(tensors, grid, fwhm=jnp.array([1.0, 10.0]))
    narrow = float(jnp.max(y[grid < 0]))    # site at delta = -20 -> grid < 0
    broad = float(jnp.max(y[grid > 0]))     # site at delta = +20
    assert narrow > broad  # same area, smaller width => taller peak


def test_bad_per_site_shape_raises():
    grid = nmr_spectra.make_grid(-10.0, 10.0, 64)
    tensors = jnp.zeros((2, 3, 3))
    with pytest.raises(ValueError):
        nmr_spectra.mas_spectrum(tensors, grid, fwhm=jnp.array([1.0, 2.0, 3.0]))


# ── powder ────────────────────────────────────────────────────────────────────


def test_powder_of_an_isotropic_tensor_reduces_to_a_single_line():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    tensors = jnp.asarray((25.0 * np.eye(3, dtype=np.float32))[None])
    powder = nmr_spectra.powder_spectrum(tensors, grid, fwhm=3.0, n_orientations=512)
    mas = nmr_spectra.mas_spectrum(tensors, grid, fwhm=3.0)
    np.testing.assert_allclose(np.asarray(powder), np.asarray(mas), atol=1e-5)


def test_powder_centre_of_mass_is_the_isotropic_shift():
    """Powder averaging redistributes intensity but conserves the first moment."""
    grid = nmr_spectra.make_grid(-400.0, 400.0, 40_001)
    tensors = jnp.asarray(_tensor(30.0, aniso=60.0, asym=0.3)[None])
    y = nmr_spectra.powder_spectrum(tensors, grid, fwhm=2.0, n_orientations=2048)
    centre = float(jnp.trapezoid(y * grid, grid) / jnp.trapezoid(y, grid))
    assert centre == pytest.approx(-30.0, abs=0.3)


def test_powder_pattern_is_rotation_invariant():
    """A powder contains every orientation, so rotating the crystal changes nothing.

    This is the physical check that sigma(n) = n.sigma.n is being averaged correctly.
    """
    grid = nmr_spectra.make_grid(-200.0, 200.0, 2001)
    base = _tensor(20.0, aniso=50.0, asym=0.5)
    rot = _random_rotation(seed=3)
    rotated = rot @ base @ rot.T

    kwargs = dict(fwhm=6.0, n_orientations=4096)
    y_base = nmr_spectra.powder_spectrum(jnp.asarray(base[None]), grid, **kwargs)
    y_rot = nmr_spectra.powder_spectrum(jnp.asarray(rotated[None]), grid, **kwargs)

    peak = float(jnp.max(y_base))
    np.testing.assert_allclose(np.asarray(y_base), np.asarray(y_rot), atol=0.03 * peak)


def test_powder_spans_the_principal_values():
    """Intensity must live between the extreme eigenvalues and nowhere else."""
    grid = nmr_spectra.make_grid(-200.0, 200.0, 4001)
    tensor = _tensor(0.0, aniso=60.0, asym=0.0)      # eigenvalues -30, -30, +60
    y = nmr_spectra.powder_spectrum(
        jnp.asarray(tensor[None]), grid, fwhm=1.0, n_orientations=4096
    )
    support = grid[y > 0.02 * jnp.max(y)]
    # referenced with delta = -sigma, so the support maps to [-60, +30]
    assert float(jnp.min(support)) == pytest.approx(-60.0, abs=3.0)
    assert float(jnp.max(support)) == pytest.approx(30.0, abs=3.0)


def test_powder_only_sees_the_symmetric_part():
    grid = nmr_spectra.make_grid(-200.0, 200.0, 2001)
    base = _tensor(10.0, aniso=40.0)
    antisym = np.array([[0.0, 5.0, -3.0], [-5.0, 0.0, 2.0], [3.0, -2.0, 0.0]], dtype=np.float32)

    kwargs = dict(fwhm=5.0, n_orientations=1024)
    y_sym = nmr_spectra.powder_spectrum(jnp.asarray(base[None]), grid, **kwargs)
    y_full = nmr_spectra.powder_spectrum(jnp.asarray((base + antisym)[None]), grid, **kwargs)
    np.testing.assert_allclose(np.asarray(y_sym), np.asarray(y_full), atol=1e-5)


def test_powder_area_equals_total_weight():
    grid = nmr_spectra.make_grid(-400.0, 400.0, 20_001)
    tensors = jnp.asarray(np.stack([_tensor(10.0, aniso=40.0), _tensor(-30.0, aniso=20.0)]))
    y = nmr_spectra.powder_spectrum(tensors, grid, fwhm=3.0, n_orientations=1024)
    assert float(jnp.trapezoid(y, grid)) == pytest.approx(2.0, rel=1e-3)


# ── graph interface ───────────────────────────────────────────────────────────


def _graph(numbers, tensors, field=keys.NMR_TENSORS, mask=None):
    nodes = {
        keys.ATOMIC_NUMBERS: jnp.asarray(numbers),
        field: jnp.asarray(tensors),
    }
    if mask is not None:
        nodes[keys.MASK] = jnp.asarray(mask)
    n = len(numbers)
    return jraph.GraphsTuple(
        nodes=nodes,
        edges={},
        globals={},
        senders=jnp.zeros((0,), dtype=int),
        receivers=jnp.zeros((0,), dtype=int),
        n_node=jnp.array([n]),
        n_edge=jnp.array([0]),
    )


def test_spectrum_from_graph_selects_one_element():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 4001)
    graph = _graph([11, 14], np.stack([_tensor(10.0), _tensor(60.0)]))

    na = nmr_spectra.spectrum_from_graph(graph, grid, atomic_number=11, fwhm=2.0)
    si = nmr_spectra.spectrum_from_graph(graph, grid, atomic_number=14, fwhm=2.0)
    both = nmr_spectra.spectrum_from_graph(graph, grid, fwhm=2.0)

    assert float(grid[int(jnp.argmax(na))]) == pytest.approx(-10.0, abs=0.2)
    assert float(grid[int(jnp.argmax(si))]) == pytest.approx(-60.0, abs=0.2)
    np.testing.assert_allclose(np.asarray(both), np.asarray(na + si), atol=1e-6)


def test_spectrum_from_graph_excludes_padding_nodes():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 4001)
    real = _graph([11], _tensor(10.0)[None])
    padded = _graph(
        [11, 11], np.stack([_tensor(10.0), _tensor(60.0)]), mask=[True, False]
    )
    y_real = nmr_spectra.spectrum_from_graph(real, grid, atomic_number=11, fwhm=2.0)
    y_padded = nmr_spectra.spectrum_from_graph(padded, grid, atomic_number=11, fwhm=2.0)
    np.testing.assert_allclose(np.asarray(y_real), np.asarray(y_padded), atol=1e-6)


def test_spectrum_from_graph_missing_field_raises():
    graph = _graph([11], _tensor(1.0)[None])
    with pytest.raises(KeyError):
        nmr_spectra.spectrum_from_graph(
            graph, nmr_spectra.make_grid(-1.0, 1.0, 8), field="not_a_field"
        )


# ── comparison / loss ─────────────────────────────────────────────────────────


def test_identical_spectra_give_zero_loss():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    y = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(15.0)[None]), grid, fwhm=4.0)
    assert float(nmr_spectra.spectrum_loss(y, y, x=grid)) == pytest.approx(0.0, abs=1e-6)


def test_loss_is_blind_to_an_overall_scale():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    y = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(15.0)[None]), grid, fwhm=4.0)
    for kwargs in (
        dict(normalise_mode="area"),
        dict(normalise_mode="none", metric="cosine"),
        dict(normalise_mode="none", fit_scale=True),
    ):
        loss = nmr_spectra.spectrum_loss(y, 7.3 * y, x=grid, **kwargs)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_loss_grows_with_peak_separation():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    target = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(0.0)[None]), grid, fwhm=4.0)
    losses = [
        float(
            nmr_spectra.spectrum_loss(
                nmr_spectra.mas_spectrum(jnp.asarray(_tensor(d)[None]), grid, fwhm=4.0),
                target,
                x=grid,
            )
        )
        for d in (1.0, 5.0, 20.0)
    ]
    assert losses[0] < losses[1] < losses[2]


def test_resample_puts_an_experimental_axis_on_the_working_grid():
    x_src = jnp.linspace(-50.0, 50.0, 101)
    y_src = nmr_spectra.gaussian(x_src, 10.0, 8.0)
    grid = nmr_spectra.make_grid(-50.0, 50.0, 401)
    y = nmr_spectra.resample(x_src, y_src, grid)
    assert float(grid[int(jnp.argmax(y))]) == pytest.approx(10.0, abs=0.5)
    # outside the measured range there is no data
    assert float(nmr_spectra.resample(x_src, y_src, jnp.array([-200.0]))[0]) == 0.0


@pytest.mark.parametrize("mode", ["area", "max"])
def test_normalise_modes(mode):
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    y = 5.0 * nmr_spectra.mas_spectrum(jnp.asarray(_tensor(0.0)[None]), grid, fwhm=4.0)
    n = nmr_spectra.normalise(y, mode, grid)
    if mode == "max":
        assert float(jnp.max(n)) == pytest.approx(1.0, rel=1e-6)
    else:
        assert float(jnp.trapezoid(n, grid)) == pytest.approx(1.0, rel=1e-3)


# ── differentiability (the reason the module exists) ──────────────────────────


@pytest.mark.parametrize("mode", ["mas", "powder"])
def test_spectrum_loss_is_differentiable_wrt_the_tensors(mode):
    grid = nmr_spectra.make_grid(-100.0, 100.0, 1001)
    target_t = jnp.asarray(_tensor(20.0, aniso=30.0)[None])
    start_t = jnp.asarray(_tensor(26.0, aniso=30.0)[None])
    spectrum = (
        nmr_spectra.mas_spectrum
        if mode == "mas"
        else lambda t, g, **kw: nmr_spectra.powder_spectrum(t, g, n_orientations=256, **kw)
    )
    target = spectrum(target_t, grid, fwhm=5.0)

    def loss(tensors):
        return nmr_spectra.spectrum_loss(spectrum(tensors, grid, fwhm=5.0), target, x=grid)

    value, grad = jax.value_and_grad(loss)(start_t)
    assert np.isfinite(float(value)) and float(value) > 0.0
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.linalg.norm(grad)) > 1e-8, "gradient vanished — loss is flat"


def test_gradient_points_downhill():
    """A step along -grad must lower the loss: the signal a structure-resolution
    optimiser relies on."""
    grid = nmr_spectra.make_grid(-100.0, 100.0, 1001)
    target = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(20.0)[None]), grid, fwhm=5.0)

    def loss(tensors):
        return nmr_spectra.spectrum_loss(
            nmr_spectra.mas_spectrum(tensors, grid, fwhm=5.0), target, x=grid
        )

    start = jnp.asarray(_tensor(26.0)[None])
    value, grad = jax.value_and_grad(loss)(start)
    stepped = float(loss(start - 0.5 * grad / (jnp.linalg.norm(grad) + 1e-30)))
    assert stepped < float(value)


def test_spectrum_functions_are_jittable():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 512)
    tensors = jnp.asarray(_tensor(10.0, aniso=20.0)[None])
    mas = jax.jit(lambda t: nmr_spectra.mas_spectrum(t, grid, fwhm=3.0))(tensors)
    powder = jax.jit(
        lambda t: nmr_spectra.powder_spectrum(t, grid, fwhm=3.0, n_orientations=128)
    )(tensors)
    assert np.all(np.isfinite(np.asarray(mas)))
    assert np.all(np.isfinite(np.asarray(powder)))


# ── Wasserstein: the metric that survives non-overlapping peaks ───────────────


def test_wasserstein_measures_peak_separation_in_ppm():
    grid = nmr_spectra.make_grid(-200.0, 200.0, 8001)
    a = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(0.0)[None]), grid, fwhm=4.0)
    for sep in (5.0, 20.0, 60.0):
        b = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(sep)[None]), grid, fwhm=4.0)
        w = float(nmr_spectra.spectrum_loss(a, b, x=grid, metric="wasserstein"))
        assert w == pytest.approx(sep, rel=0.02)


def test_pointwise_metrics_saturate_but_wasserstein_does_not():
    """The reason `wasserstein` exists: once two narrow lines stop overlapping, a
    point-by-point loss is constant and its gradient vanishes."""
    grid = nmr_spectra.make_grid(-300.0, 300.0, 6001)
    target = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(0.0)[None]), grid, fwhm=4.0)

    def losses(sep, metric):
        y = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(sep)[None]), grid, fwhm=4.0)
        return float(nmr_spectra.spectrum_loss(y, target, x=grid, metric=metric))

    far, farther = losses(40.0, "rmse"), losses(120.0, "rmse")
    assert far == pytest.approx(farther, rel=1e-6)      # saturated: no signal left

    w_far, w_farther = losses(40.0, "wasserstein"), losses(120.0, "wasserstein")
    assert w_farther > 2.5 * w_far                      # still growing


def test_wasserstein_keeps_a_gradient_where_rmse_has_none():
    grid = nmr_spectra.make_grid(-300.0, 300.0, 6001)
    target = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(0.0)[None]), grid, fwhm=4.0)

    def make_loss(metric):
        def loss(iso):
            tensors = iso * jnp.eye(3)[None]
            y = nmr_spectra.mas_spectrum(tensors, grid, fwhm=4.0)
            return nmr_spectra.spectrum_loss(y, target, x=grid, metric=metric)
        return loss

    far = jnp.float32(80.0)   # 20 linewidths away
    g_rmse = float(jnp.abs(jax.grad(make_loss("rmse"))(far)))
    g_w = float(jnp.abs(jax.grad(make_loss("wasserstein"))(far)))
    assert g_rmse < 1e-6
    assert g_w > 0.5


def test_wasserstein_is_zero_for_identical_spectra_and_scale_free():
    grid = nmr_spectra.make_grid(-100.0, 100.0, 2001)
    y = nmr_spectra.mas_spectrum(jnp.asarray(_tensor(15.0)[None]), grid, fwhm=4.0)
    assert float(nmr_spectra.spectrum_loss(y, y, x=grid, metric="wasserstein")) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(
        nmr_spectra.spectrum_loss(y, 9.1 * y, x=grid, metric="wasserstein")
    ) == pytest.approx(0.0, abs=1e-6)
