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
        # pylint: disable=attribute-defined-outside-init
        
        # 1. Definizione della Funzione di Differenziazione (Derivata Prima)
        self._diff_fn = gcnn.experimental.diff(
            self.B_ind_fn,
            # OUTPUT (funzione di cui calcolare la derivata): Campo Magnetico Indotto
            # f"nodes.{self.B_ind}:Iγ"  indica:
            # - 'nodes': Il campo è per-nodo.
            # - 'I': Etichetta di contrazione per l'INDICE DEL NODO (N_atomi). ESSENZIALE!
            # - 'γ': Etichetta di contrazione per la direzione CARTESIANA (3D).
            f"nodes.{self.B_ind}:Iγ",
            
            wrt=[
                # INPUT (rispetto a cui differenziare): Campo Magnetico Esterno
                # f"globals.{self.B_ext}:gα" indica:
                # - 'globals': Il campo è globale (o batchato).
                # - 'g': Etichetta di contrazione per l'INDICE GLOBALE/BATCH (B).
                # - 'α': Etichetta di contrazione per la direzione CARTESIANA (3D).
                f"globals.{self.B_ext}:gα",
            ],
            
            # OUTPUT (il risultato della derivazione)
            # ":Iγα" indica che il risultato deve avere le dimensioni delle etichette non 'contratte' (non sommate)
            # relative all'output (I, γ) e all'input (α).
            # Risultato atteso: (I, γ, α) -> (N_atomi, 3, 3) se il batch è gestito separatamente o B=1.
            out=":Iγα",
            
            return_graph=True,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # 1. Prepara l'Input per la Traccia
        # B_ext_zeros è necessario per la tracciatura JAX. Il valore effettivo del campo esterno
        # (che dovrebbe essere zero o un valore di test) è già implicitamente gestito dal Jacobiano/diff.
        B_ext_zeros = jnp.zeros_like(graph.globals[self.B_ext])

        # 2. Esecuzione della Derivazione
        # La funzione differenziata restituisce due valori: la derivata e il grafo risultante.
        # Devi fornire gli input specificati in 'wrt' (in questo caso solo B_ext).
        derivative, graph = self._diff_fn(
            graph,
            B_ext_zeros,
        )
        
        # 3. Applicazione del Segno e Riassegnazione
        # Il tensore di shielding è definito con un segno negativo.
        # N.B.: Con gcnn.diff correttamente etichettato, NON è necessaria alcuna operazione di sum/squeeze.
        shielding = -derivative
        
        # 4. Aggiornamento del Grafo
        # Usiamo l'approccio fluente (update_graph) per inserire il tensore di shielding
        # nei nodi del grafo, sotto la chiave 'predicted(keys.NMR_TENSORS)'.
        graph = (
            gcnn.experimental.update_graph(graph)
            .set(("nodes", self.out_key), shielding)
            .get()
        )
        
        # print("graph nodes keys:", graph.nodes.keys())
        # print("graph globals keys:", graph.globals.keys())
        # print("predicted_induced_magnetic_field:", graph.nodes["predicted_induced_magnetic_field"])

        # print("external magnetic field shape:", graph.globals["external_magnetic_field"].shape)

        # print("predicted induced magnetic field shape:", graph.nodes["predicted_induced_magnetic_field"].shape)

        # print("NMR tensors shape:", graph.nodes["NMR_tensors"].shape)
        # print("predicted NMR tensors shape:", graph.nodes["predicted_NMR_tensors"].shape)


        return graph
