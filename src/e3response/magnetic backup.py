from collections.abc import Callable
from typing import Union

from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted
import tensorial.typing as tt

from . import keys

__all__ = "MagneticShieldingTensor"


class MagneticShieldingTensor(linen.Module):
    """
    flax.linen.Module for computing magnetic shielding tensors σ_{k, ij}
    for each atom k, based on the linear response of the induced magnetic field
    B^{k}_ind to an applied external magnetic field B_ext.

    The Jacobian is computed as:
        σ_{k, ij} = - ∂B_ind_{k, i} / ∂B_ext_j

    Returns:
        A graph where each node has a (3, 3) tensor stored in `out_field`.
    """

    B_ind_fn: gcnn.GraphFunction
    B_ind: str = predicted(keys.INDUCED_MAGNETIC_FIELD)
    B_ext: str = keys.EXTERNAL_MAGNETIC_FIELD
    out_key: str = predicted(keys.NMR_TENSORS)

    def setup(self) -> None:

        self._jacobian_fn = gcnn.jacobian(
            of=f"nodes.{self.B_ind}",
            wrt=f"globals.{self.B_ext}",
            has_aux=True,
            sum_axis=False,
        )(self.B_ind_fn)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        B_ext_zeros = jnp.zeros_like(graph.globals[self.B_ext])
        shielding, graph = self._jacobian_fn(graph, B_ext_zeros)

        # *** INSERISCI QUESTO ***
        print("Shape di shielding dopo il Jacobiano:", shielding.shape) 
        
        shielding = -shielding.sum(2) 
        
        # *** INSERISCI QUESTO ***
        print("Shape di shielding dopo la somma:", shielding.shape)

        # updates = gcnn.utils.UpdateGraphDicts(graph)
        # updates.nodes[self.out_field] = shielding

        # graph = updates.get()

        graph = (
            gcnn.experimental.update_graph(graph)
            .set(("nodes", self.out_key), shielding) 
            .get()
            )       
        
        # print("graph nodes keys:", graph.nodes.keys())
        # print("graph globals keys:", graph.globals.keys())
        # print("predicted_induced_magnetic_field:", graph.nodes["predicted_induced_magnetic_field"])

        print("external magnetic field shape:", graph.globals["external_magnetic_field"].shape)

        print("predicted induced magnetic field shape:", graph.nodes["predicted_induced_magnetic_field"].shape)

        print("NMR tensors shape:", graph.nodes["NMR_tensors"].shape)
        print("predicted NMR tensors shape:", graph.nodes["predicted_NMR_tensors"].shape)


        return graph
