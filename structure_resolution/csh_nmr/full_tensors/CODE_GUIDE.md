# `structure_resolution_grad_nmr.py` — code guide (CSH / Na)

Reference for the CSH version of the gradient-based NMR structure-resolution demo.
Line numbers refer to `structure_resolution_grad_nmr.py` in this directory.

---

## 1. What the script asks

> Given the NMR shielding tensors of a structure, can we recover the position of one
> displaced atom by minimising the mismatch between predicted and target tensors?

The experiment is a controlled inversion test with a known answer:

1. Take a CSH structure whose true geometry and DFT tensors are known.
2. Displace one atom (here Na) by a random Gaussian kick — we now "don't know" where it was.
3. Minimise `RMSE(predicted tensors, target tensors)` over that atom's 3 coordinates.
4. Compare the recovered position with the true one. `|Δr| < DIST_THRESHOLD` = success.

The key enabling fact: the model recomputes edge vectors from `nodes.positions`, so the
loss is differentiable end-to-end and JAX gives the **exact** `∂loss/∂pos_k`. No finite
differences, no surrogate.

Two independent things are being tested, and they are worth keeping apart:

| Question | How it is probed |
| --- | --- |
| Does the *optimiser* work? | `LOSS_TARGET="model"` — target = the model's own prediction at the true geometry, so loss@true ≡ 0 by construction. Any failure is the optimiser or the landscape. |
| Is the *model* good enough? | `LOSS_TARGET="dft"` — target = DFT labels. loss@true is now the model's own error floor (~3–6 ppm here), and the minimum may not sit at the true geometry. |

Run the `"model"` self-consistency test first. If recovery fails there, nothing about the
`"dft"` numbers is interpretable.

---

## 2. Execution order at a glance

The file is a **script**, not a module: everything at module level runs top to bottom.
Helpers are defined first, then the numbered STEP blocks execute.

```
CONFIGURATION            (50)    all knobs, edited in place
HELPERS                  (141)   pure functions, no side effects
STEP 0   Load model      (585)   checkpoint → module, params, cfg
STEP 0b  Reproduce split (596)   cfg.data → the run's own test split
STEP 1   Ground truth    (619)   pick structure + atom k, build target tensors
STEP 2   Perturbation    (671)   displace atom k
STEP 3   Loss + sanity   (683)   build differentiable loss, verify it is not flat
STEP 4   Optimisation    (989)   BFGS on 3 DOF, optional multistart
STEP 5   Evaluation      (1016)  plot, CSV, GIF
STEP 6   Statistics      (1136)  repeat over many structures
```

---

## 3. Configuration block (line 50)

### Model and data provenance

| Knob | Meaning |
| --- | --- |
| `RUN_DIR` | Checkpoint dir; must hold `config.yaml` + `checkpoints/last.ckpt`. **Everything about the data comes from this config** (`r_max`, `limit`, `train_val_test_split`), which is what makes the reproduced split match the trained run. |
| `DATA_FILE` | Overrides only the JSON path, since the run's config may store a cluster path. |
| `SPLIT` | `"train"` / `"val"` / `"test"` / `"full"`. Use `"test"` — held-out data is the honest test. |
| `LIMIT` | Restriction applied **before** splitting. Leave `None` so the run's own `limit` (`"bulk"` for this checkpoint) is used — changing it changes the split and breaks reproduction. |
| `SPLIT_LIMIT` | Restriction applied **after** splitting, i.e. "just the first N test structures". Safe to change; does not perturb the split. |

### Case selection

| Knob | Meaning |
| --- | --- |
| `MOL_IDX` | Index into the (limited) split, not the raw dataset. `None` → random pick from the seeded RNG. |
| `ATOM_K` | `None` → random atom; `int` → that index; `str` → random atom of that species (`"Na"`). |
| `SEED` | Seeds structure choice, atom choice and the perturbation. Reproducible. |

### Perturbation

| Knob | Meaning |
| --- | --- |
| `SIGMA` | **Std-dev per Cartesian component**, not a maximum. `|Δr|` follows a chi distribution with 3 dof: mean ≈ `1.60·σ`, median ≈ `1.54·σ`, mode ≈ `1.41·σ`, and the tail is unbounded (σ=0.5 → mean 0.80 Å, 95th pct 1.40 Å, occasional 2.3 Å). |
| `MIN_DIST` | Rejection radius: a draw landing closer than this to any other atom (minimum-image) is redrawn. Does **not** cap how far the atom can go. |
| `MAX_PERTURB_ATTEMPTS` | Retry cap for the rejection sampler. At σ=0.5 / `MIN_DIST=0.2` the rejection rate is ~0 %, so this is effectively never hit. |

### Loss

| Knob | Meaning |
| --- | --- |
| `LOSS_TARGET` | `"dft"` or `"model"` — see the table in §1. |
| `NEIGHBOR_CUTOFF` | Radius (Å) around atom k whose tensors enter the loss. **Minimum-image**, so cross-boundary neighbours count. Tune to the Na coordination shell. |

### Optimiser

| Knob | Meaning |
| --- | --- |
| `MAX_ITER` | BFGS iteration cap (`status=1` in the output means it stopped here). |
| `GRAD_CHECK` | One-off finite-difference verification of the JAX gradient. |
| `PRINT_EVALS` | Print every evaluation, line-search probes included. Noisy inside STEP 6. |
| `MAX_DISPLACEMENT` | Radius `R` of the smooth tanh cap on atom k's excursion from its start. Must exceed the drawn displacement to permit recovery — ~2–3× SIGMA is the useful range. Very large values (e.g. 20 Å) disable the guard, which is what it exists to prevent. `None` → raw unbounded BFGS. |
| `N_RESTARTS` | Max BFGS runs per atom. `0` or `1` = a single run. Extra runs only fire while the best loss is still above `RESTART_LOSS_TOL`. |
| `RESTART_LOSS_TOL` | "Good enough" loss (ppm) → stop restarting. Also the loss ceiling for a STEP 6 recovery to count as *reliable*. Keep near the model's val RMSE. |
| `RESTART_JITTER` | σ of the Gaussian jitter seeding each restart. |

### Statistics and output

`RUN_STATS`, `N_MOL_STAT`, `N_ATOM_STAT`, `DIST_THRESHOLD` drive STEP 6.
`SAVE_CSV` / `SAVE_GIF` / `SAVE_TENSOR_GIF` / `GIF_MAX_FRAMES` / `GIF_FPS` drive STEP 5.
Every run writes to `./results/<timestamp>/`, so nothing is ever overwritten.

---

## 4. Helpers (line 141)

### Model and graph plumbing

**`load_model(run_dir)` (143)** — loads `config.yaml`, instantiates the model via hydra,
restores `checkpoints/last.ckpt`, and converts the checkpoint's numpy parameters to JAX
arrays (needed for `jit`/`grad`). Returns `(module, params, cfg)`.

**`make_graph_builder(r_max)` (156)** — returns `atoms → jraph.GraphsTuple` with the same
fields as `CshNmrDataModule._to_graph`: `atom_include_keys=("numbers", "nmr_tensors")`,
globals `(external_magnetic_field, energy)`. Two properties matter:

- The **neighbour list is rebuilt from scratch on every call** — mandatory, because atom k
  moves and the topology can change.
- Since the Atoms carry a cell and `pbc=True`, `graph_from_ase` builds the neighbour list
  **with periodic images**. Periodicity is handled for free here; the *host-side* distance
  checks are the parts that needed explicit `mic=True` (see below).

**`make_predictor(module, params, to_graph)` (169)** — returns `predict(atoms) → (n_atoms, 3, 3)`.
Rebuilds the graph, applies the jitted model, slices to the real atoms. Used for the host-side
sanity checks and plots, *not* inside the optimiser. Recompiles whenever the edge count changes.

### Geometry

**`min_image_distances(atoms, atom_k, pos_k=None, others_only=False)` (190)** — distances from
atom k to all atoms under the minimum-image convention. `pos_k` moves atom k first (on a copy);
`others_only` drops k's own zero entry. Falls back to plain distances when `pbc` is all-False,
so it is also correct for a molecule. **This is the periodic-solid correction**: without it an
atom near a cell face looks many Å from a neighbour that is actually 0.5 Å away through the
boundary. On a test structure the 5 Å shell around Na contains **35 atoms with mic vs 13 without**.

**`perturb_atom(atoms, k, sigma, min_dist, rng, max_attempts)` (211)** — rejection sampler.
Each attempt is an **independent fresh draw from the original position** (not a cumulative
walk); it returns at the first draw whose minimum-image distance to every other atom is
≥ `min_dist`, and raises `RuntimeError` if all attempts fail (STEP 6 catches it and skips).

**`select_atom_k(atoms, n_atoms, cfg, rng)` (234)** — resolves `ATOM_K` (None / int / species
string) to an index; returns `None` if the species is absent.

**`neighbor_mask(atoms, atom_k, cutoff)` (252)** — boolean mask of atoms within `cutoff` of k,
k included, minimum-image. Selects the atoms whose tensors enter the loss.

**`compute_loss(pred, target, mask)` (260)** — host-side masked RMSE in ppm, for the STEP 3
sanity check and prints. Same formula as the traced loss, so numbers are comparable.

### Loss and gradient

**`minimize_fn(fun, what, wrt)` (699)** — wraps a graph function with `gcnn.adapt` so the
optimised quantity is injected at `wrt` (here `globals.pos_k`) and the scalar at `what`
(`globals.loss`) is returned, then hands it to `jax.scipy.optimize.minimize`. The whole
minimisation runs **on device**. This is the canonical repo idiom, reproduced verbatim.

**`_bound_disp(u, x0, max_disp, xp)` (717)** — the displacement cap. With `d = u − x0`,
`r = |d|`, it returns `x0 + d · (R·tanh(r/R)/r)`, so `|pos − x0| = R·tanh(r/R) < R` for *any*
`u`, while small steps (`r ≪ R`) pass through nearly unchanged. Smooth everywhere, so BFGS
still sees a differentiable objective. `xp` is `jnp` inside the traced loss and `np` on the
host — the same map is used to convert the optimiser's answer back to a real position.

**`make_value_and_grad(...)` (734)** — `jax.value_and_grad` of the loss w.r.t. the optimiser
variable, taking `(graph, u)`. Injects `pos_k` into `nodes.positions`, runs the model, computes
the masked RMSE, and carries the predicted σ[k] as aux. Used only by `_grad_check`.

**`make_loss_graph_fn(...)` (763)** — the loss actually minimised. Reads `u` from
`globals.pos_k`, maps it through `_bound_disp`, writes it into `nodes.positions`, runs the
model, and stores the masked RMSE at `globals.loss`. If a `recorder` is given, a
`jax.debug.callback(..., ordered=True)` fires on the host at **every** evaluation
(line-search probes included) with the *actual bounded* `pos_k` — this is what makes the
trajectory observable even though `jopt` exposes no per-iteration hook. The callback is
transparent to autodiff, so gradients are unaffected.

Loss definition in both cases:

```
loss = sqrt( Σ_{i ∈ nbr}  ‖pred_i − target_i‖²_F  /  (9 · n_nbr) )     [ppm]
```

i.e. RMSE over all 9 components of the selected atoms' tensors — same units as the training
metric `val/nmr_tensors_rmse`, so the two are directly comparable.

**`_grad_check(vg, to_graph, atoms_base, atom_k, x0, eps)` (810)** — central differences vs
the analytic gradient, both on **one fixed graph** so a topology change cannot corrupt the
comparison. Interpretation: `|JAX| ≈ 0` means autodiff never reached `pos_k` (a real bug);
rel.err up to ~0.2 is just float32 FD noise on a steep landscape.

### Optimiser driver

**`optimize_position(...)` (855)** — the orchestrator.

- `_jittered_start()` — a valid restart seed: `x0` + Gaussian jitter, rejected if it clashes
  (minimum-image) with another atom.
- `_single_run(start, run_idx)` — builds the **fixed-topology** graph at `start`, attaches
  `globals.pos_k`, runs BFGS through `minimize_fn`, then maps `result.x` back through
  `_bound_disp` so callers read a real position, not the unconstrained variable.
- Multistart loop — always at least one run (`range(max(1, N_RESTARTS))`), keeps the run with
  the **lowest final loss**, breaks early once a run beats `RESTART_LOSS_TOL`. Selection is by
  loss alone, never by `|Δr|`: in a real deployment the true position is unknown.

Returns `dict(result, traj, n_evals, n_runs)`.

### Plot and IO helpers

| Helper | Output |
| --- | --- |
| `save_gif` (271) | 2×2 animation: 3-D trajectory of atom k (species-coloured atoms, true position as a star, start as a triangle), loss curve, distance-to-true curve, and the σ[k] ellipsoid. Non-finite losses from bad line-search probes are filtered out of the axis limits and forward-filled in the ellipsoid. |
| `_ellipsoid_xyz` (416) | Tensor → ellipsoid surface. Symmetrises, eigendecomposes, semi-axes = `|eigenvalues| / norm`, rotated into the eigenframe. Shared normalisation so target and prediction are comparable. |
| `save_tensor_gif` (435) | Standalone predicted-vs-target ellipsoid animation (green = target, blue = predicted). |
| `save_statistics_figure` (492) | STEP 6's four panels: **(1)** ECDF of `|Δr|` — the success curve; **(2)** final vs initial `|Δr|` on log axes, below `y=x` means the atom improved; **(3)** loss vs `|Δr|` — the *identifiability* panel, where low loss at high `|Δr|` marks a degenerate landscape; **(4)** `|Δr|` histogram with the threshold and success rate. |

---

## 5. The STEP blocks

### STEP 0 — Load model (585)

`load_model(RUN_DIR)`, then `to_graph = make_graph_builder(cfg.data.r_max)` and
`predictor = make_predictor(...)`. `r_max` comes from the run's config, never hard-coded.

### STEP 0b — Reproduce the run's split (596)

`reax.data.random_split` draws from the Engine's default `nnx.Rngs(0)` stream, which nothing
in this repo overrides. `CshNmrDataModule.load_split_structures()` replays the same
struct_type-grouped stratified split and returns the partition as **ASE Atoms** — no saved
seed or index list needed, as long as the data file and `limit` are unchanged.

Atoms, not graphs, because the demo *moves* an atom: a `GraphsTuple` has its topology frozen
at construction. (`SPLIT="full"` bypasses splitting via `_load_structures()`.)

### STEP 1 — Ground truth (619)

Picks the structure (random if `MOL_IDX is None`) and atom k, builds `nbr_mask`, reads the DFT
tensors from the graph, and evaluates the model at the true geometry. Then selects the target
per `LOSS_TARGET` and prints the model-vs-DFT RMSE — a first look at whether this structure is
even within the model's competence.

### STEP 2 — Perturbation (671)

Draws the displacement and reports `|Δpos|`, the true position and the perturbed one.

### STEP 3 — Loss + sanity check (683)

Builds the loss, then checks the landscape is not flat: **`loss@perturbed` must be clearly
above `loss@true`**. If not, the tensors of atom k and its neighbours barely depend on k's
position, no optimiser can help, and this is physics rather than a bug. Hard failure in
`"dft"` mode (`AssertionError`), warning only in `"model"` mode where loss@true ≈ 0.

### STEP 4 — Optimisation (989)

Runs `optimize_position` from the perturbed position and reports the loss trajectory,
`|Δr|` before/after, and the verdict. Success is decided by **positional error**
(`rmsd < DIST_THRESHOLD`), never by the optimiser's convergence flag.

`jax.scipy.optimize`'s BFGS `status` reads: `0` converged, `1` `maxiter` reached,
`≥ 2` line-search failure (`2 + line_search_status`), `-1` undefined. A line-search failure
is common on a steep float32 landscape and does **not** by itself mean the recovered geometry
is bad — always read `loss_final` and `|Δr|` instead.

### STEP 5 — Evaluation and plots (1016)

Three panels: loss vs evaluation (log y, with loss@true as a reference line), distance-to-true
vs evaluation, and a bar chart comparing isotropic σ over the neighbour atoms for
target / perturbed / recovered. Then the trajectory CSV (per-evaluation position, loss,
distance), the one-row summary CSV, and optionally the GIFs.

### STEP 6 — Statistics (1136)

Loops over the structures **after** `MOL_IDX` (`stat_start = MOL_IDX + 1`, so the detailed
case is not double-counted), picking `N_ATOM_STAT` atoms per structure and running the same
pipeline. Two rates are tracked, and the distinction is the point:

- **success** — `|Δr| < DIST_THRESHOLD`. Needs ground truth, so it is unavailable in a real
  deployment.
- **reliable** — success **and** `loss_final < RESTART_LOSS_TOL`. Uses only observable
  quantities, and is therefore the honest proxy for how this would behave on real spectra.

---

## 6. Invariants worth remembering

**Fixed topology per run.** The graph is built once at the start position and only
`nodes.positions` changes during the minimisation. The gradient is therefore exact *for that
topology*. If atom k moves far, the true neighbour list would differ — which is one reason
`MAX_DISPLACEMENT` and the multistart exist rather than one long unbounded run.

**Periodicity is split across two layers.** Graph construction gets it from the cell
automatically; every host-side distance (clash checks, neighbour selection) needs
`min_image_distances`. Mixing the two conventions is silent, not loud.

**No `mask` node field.** `gcnn.keys.MASK == "mask"`, i.e. tensorial's node-*padding* mask
consumed by `NequipLayer`. CSH is fully NMR-active, so the field was removed from
`csh_nmr.py` rather than renamed. Two consequences: the batcher's padding mask is no longer
shadowed, and single-graph inference no longer mixes a numpy mask with traced JAX features
(the `"Cannot mix numpy and jax arrays"` error).

**Only the loss is observable.** Multistart selection, `RESTART_LOSS_TOL` and the *reliable*
rate all use the loss alone. `|Δr|` exists only because this is a synthetic test; real
structure resolution has spectra and nothing else.

**`²³Na is quadrupolar (spin 3/2).** The loss targets the shielding tensor. That is
self-consistent in `"model"` mode, but before comparing against experiment, check that the
observable matches — the real measurement is often the EFG / quadrupolar coupling.

---

## 7. Outputs

Everything lands in `results/<YYYYmmdd_HHMMSS>/`:

| File | Contents |
| --- | --- |
| `structure_resolution_result.png` | STEP 5 three-panel figure |
| `structure_resolution_trajectory.csv` | per-evaluation step, loss, position, distance to true |
| `structure_resolution_summary.csv` | one row: indices, displacement, evals, loss before/after, `|Δr|`, recovered, optimiser flag |
| `structure_resolution.gif` | trajectory animation (`SAVE_GIF`) |
| `tensor_ellipsoid.gif` | ellipsoid animation (`SAVE_TENSOR_GIF`) |
| `structure_resolution_statistics.png/.csv` | STEP 6 panels and per-case rows (`RUN_STATS`) |
