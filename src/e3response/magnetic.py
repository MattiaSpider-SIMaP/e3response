from flax import linen
import jax.numpy as jnp
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys

__all__ = "InducedMagneticField", "MagneticShieldingTensor"

class InducedMagneticField(linen.Module):
    """
    Flax.linen.Module for computing induced magnetic field B_ind(k) on each atom (node) k 
    via differentiation of total energy (global) with respect to the nuclear magnetic moment μ on each node.
    
    B_ind_{k, i} = ∂E_tot / ∂μ_{k, i}
    
    Returns:
    A graph where each node has a vector (3,) stored in `out_field`.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.keys.ENERGY)  
    mu_key: str = keys.NUCLEAR_MAGNETIC_MOMENT
    out_key: str = predicted(keys.INDUCED_MAGNETIC_FIELD)
    
    def setup(self) -> None:
        # Diff E against μ (per-node). No at= — shape (N_atoms, 3) unknown at setup time.
        self._diff_E_wrt_mu = gcnn.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:g",
            wrt=[f"nodes.{self.mu_key}:Iα"],
            out=":Iα",
            return_graph=True,
            mode="fwd",
        )
    
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        mu_zeros = jnp.zeros_like(graph.nodes[self.mu_key])
        
        B_ind, graph = self._diff_E_wrt_mu(
            graph,
            mu_zeros,
        )

        graph = (
            gcnn.experimental.update_graph(graph)
            .set(("nodes", self.out_key), B_ind)
            .get()
        )
        
        return graph

class MagneticShieldingTensor(linen.Module):
    """
    flax.linen.Module for computing magnetic shielding tensors σ_{k, ij}
    for each atom k, based on the linear response of the induced magnetic field
    B^{k}_ind to an applied external magnetic field B_ext.

    The Jacobian is computed as:
        σ_{k, ij} = ∂B_ind_{k, i} / ∂B_ext_j

    Returns:
        A graph where each node has a (3, 3) tensor stored in `out_field`.
    """

    B_ind_fn: gcnn.GraphFunction
    B_ind: str = predicted(keys.INDUCED_MAGNETIC_FIELD)
    B_ext: str = keys.EXTERNAL_MAGNETIC_FIELD
    out_key: str = predicted(keys.NMR_TENSORS)
    B_ext_at_graph: bool = False  # if True, evaluate Jacobian at graph.globals[B_ext]; else at zero

    def setup(self) -> None:
        self._diff_fn = gcnn.diff(
            self.B_ind_fn,
            f"nodes.{self.B_ind}:Iγ",
            wrt=[f"globals.{self.B_ext}:gα"],
            out=":Iγα",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # B_ext is always passed explicitly at call time (zeros or graph value) adding the dimension corrisonding to the number of graphs in batch.
        if self.B_ext_at_graph:
            B_ext_val = graph.globals[self.B_ext]
        else:
            n_graphs = graph.n_node.shape[0]
            B_ext_val = jnp.zeros((n_graphs, 3))

        shielding, graph = self._diff_fn(graph, B_ext_val)
        
        graph = (
            gcnn.experimental.update_graph(graph)
            .set(("nodes", self.out_key), shielding)
            .get()
        )
        return graph
