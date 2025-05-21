"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Literal

import numpy as np
import torch
from torch_geometric.utils import to_dense_batch

from diffcsp.common.data_utils import lattice_params_to_matrix_torch, frac_to_cart_coords, cart_to_frac_coords
from flowmm.cfg_utils import dataset_options
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from flowmm.geometric_ import mask_2d_to_batch
from flowmm.rfm.manifolds import (
    EuclideanWithLogProb,
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
    MultiAtomFlatDirichletSimplex,
    NullManifoldWithDeltaRandom,
    ProductManifoldWithLogProb,
    MaskedNoDriftEuclidean
)
from flowmm.rfm.manifolds.analog_bits import (
    MultiAtomAnalogBits,
    analog_bits_to_int,
    int_to_analog_bits,
)
from flowmm.rfm.manifolds.flat_torus import (
    MaskedNoDriftFlatTorus01,
    MaskedNoDriftFlatTorus01WrappedNormal,
)
from flowmm.rfm.manifolds.lattice_params import LatticeParams, LatticeParamsNormalBase
from flowmm.rfm.manifolds.spd import (
    SPDGivenN,
    lattice_params_to_spd_vector,
    spd_vector_to_lattice_matrix,
)
from flowmm.rfm.vmap import VMapManifolds

Dims = namedtuple("Dims", ["a", "f", "l"])
ManifoldGetterOut = namedtuple(
    "ManifoldGetterOut", ["flat", "flat_base", "manifold", "dims", "mask_a_or_f", "mask_f"]
)
SplitManifoldGetterOut = namedtuple(
    "SplitManifoldGetterOut",
    [
        "flat",
        "flat_base",
        "manifold",
        "a_manifold",
        "f_manifold",
        "l_manifold",
        "dims",
        "mask_a_or_f",
        "mask_f"
    ],
)
GeomTuple = namedtuple("GeomTuple", ["a", "f", "l"])
atom_type_manifold_types = Literal["null_manifold", "simplex", "analog_bits"]
coord_manifold_types = Literal[
    "flat_torus_01",
    "flat_torus_01_normal",
    "flat_torus_01_fixfirst",
    "flat_torus_01_fixfirst_normal",
]
lattice_manifold_types = Literal[
    "non_symmetric",
    "spd_euclidean_geo",
    "spd_riemanian_geo",
    "lattice_params",
    "lattice_params_normal_base",
]


class ManifoldGetter(torch.nn.Module):
    """
    Converts representations between different manifolds.
    It takes data (atom types, fractional coordinates, lattice parameters) and converts it
    into a flat representation on the manifold for subsequent processing.
    """
    def __init__(
        self,
        atom_type_manifold: atom_type_manifold_types,
        coord_manifold: coord_manifold_types,
        lattice_manifold: lattice_manifold_types,
        dataset: dataset_options | None = None,
        analog_bits_scale: float | None = None,
        length_inner_coef: float | None = None,
    ) -> None:
        super().__init__()
        self.atom_type_manifold = atom_type_manifold
        self.coord_manifold = coord_manifold
        self.lattice_manifold = lattice_manifold
        
        # For analog bits manifold, the scale must be provided
        if atom_type_manifold == "analog_bits":
            assert analog_bits_scale is not None
        self.analog_bits_scale = analog_bits_scale
        self.length_inner_coef = length_inner_coef
        self.dataset = dataset


    @property
    def predict_atom_types(self):
        """
        Returns False if using the 'null_manifold' (indicating no prediction),
        otherwise True.
        """
        return False if self.atom_type_manifold == "null_manifold" else True

    @staticmethod
    def _atomic_one_hot(a: torch.LongTensor) -> torch.LongTensor:
        """Converts atomic labels to one-hot encoding (subtracting 1 since labels start from 1)."""
        return torch.nn.functional.one_hot(a - 1, num_classes=NUM_ATOMIC_TYPES)

    @staticmethod
    def _inverse_atomic_one_hot(
        a: torch.LongTensor | np.ndarray, dim: int = -1
    ) -> torch.LongTensor:
        """
        Reverts one-hot encoding back to atomic labels.
        Handles both numpy arrays and torch tensors.
        """
        if isinstance(a, np.ndarray):
            return np.argmax(a, axis=dim) + 1
        elif isinstance(a, torch.Tensor):
            return torch.argmax(a, dim=dim) + 1
        else:
            raise TypeError()

    @staticmethod
    def _atomic_bits(a: torch.LongTensor, scale: float) -> torch.LongTensor:
        """Converts atomic labels to analog bits representation."""
        return int_to_analog_bits(a - 1, NUM_ATOMIC_BITS, scale)

    @staticmethod
    def _inverse_atomic_bits(a: torch.LongTensor | np.ndarray) -> torch.LongTensor:
        """
        Reverts analog bits representation back to atomic labels.
        Handles both numpy arrays and torch tensors.
        """
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
            return analog_bits_to_int(a).numpy() + 1
        elif isinstance(a, torch.Tensor):
            return analog_bits_to_int(a) + 1
        else:
            raise TypeError()

    @staticmethod
    def _get_max_num_atoms(mask_a_or_f: torch.BoolTensor) -> int:
        """Finds the maximum number of atoms in any graph (batch element)."""
        return int(mask_a_or_f.sum(dim=-1).max())

    @staticmethod
    def _get_num_atoms(mask_a_or_f: torch.BoolTensor) -> torch.LongTensor:
        """Counts the number of atoms per graph in the batch."""
        return mask_a_or_f.sum(dim=-1)

    def _to_dense(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        frac_coords: torch.Tensor,
        constraints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Converts sparse batch representations into dense tensors.
        Uses `to_dense_batch` to obtain dense representations of atom types and coordinates.
        """
        a, mask_a_or_f = to_dense_batch(
            x=atom_types, batch=batch
        )  # B x N x NUM_ATOMIC_TYPES, B x N
        f, _ = to_dense_batch(x=frac_coords, batch=batch)  # B x N x 3
    
        B, N = mask_a_or_f.shape
          
        val = 60
        # constraints_2d = constraints.view(B, val)  # => [B, 54]

        # if N == val:
        mask_f = constraints.bool()
        # elif N < val:
        #     mask_f = constraints_2d[:, :N].bool()
        # else:  
        #     mask_padded = torch.zeros(B, N, dtype=torch.bool, device=constraints.device)
        #     mask_padded[:, :val] = constraints_2d.bool()
        #     mask_f = mask_padded
        # mask_f = mask_a_or_f
        return a, f, mask_a_or_f, mask_f


    def _to_flat(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        l: torch.Tensor,
        dims: Dims,
    ) -> torch.Tensor:
        """
        Flattens the atomic, fractional coordinate, and lattice data
        according to their specified dimensions and concatenates them.
        """
        a_flat = a.reshape(a.size(0), dims.a)
        f_flat = f.reshape(f.size(0), dims.f)
        l_flat = l.reshape(l.size(0), dims.l)
        return torch.cat([a_flat, f_flat, l_flat], dim=1)
    
    
    
    def sample_D(self, t: torch.Tensor, shape: tuple, device=None, dtype=torch.float32) -> torch.Tensor:
        
        h_t = 1.0 * t  # h(t) = 1000 * t
        
        sigma = torch.sqrt(h_t / 1000.0)
        
        # Sample from N(0, sigma^2 I)
        return torch.normal(
            mean=torch.zeros(shape, device=device, dtype=dtype),
            std=sigma * torch.ones(shape, device=device, dtype=dtype)
        )
    
    def georep_to_flatrep(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        split_manifold: bool,
        velocities: torch.Tensor = None,
        constraints: torch.Tensor = None,
        lengths:  torch.Tensor = None,
        angles:  torch.Tensor = None,
        displace: bool = False,
        sample: bool= False
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):  
        """
        georep : raw geometric data of the crystal
        flatrep: flattened one-dimenional vector of the georep
        Converts a georepresentation (atoms, fractional coordinates, and lattices)
        to a flat manifold representation.
        Optionally splits the manifold into separate parts.
        """
    
        """converts from georep to the manifold flatrep"""
        a, f, mask_a_or_f, mask_f = self._to_dense(
            batch, atom_types=atom_types, frac_coords=frac_coords, constraints=constraints
        )
        
        num_atoms = self._get_num_atoms(mask_a_or_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            atom_types_dense_one_hot=a,
            dim_coords=frac_coords.shape[-1],
            split_manifold=split_manifold
        )
        
        flat_base = None
        if displace:
            num_atoms = self._get_num_atoms(mask_a_or_f)
            
            cart_cord = frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
            
            cart_cord += velocities
            fract_disp = cart_to_frac_coords(cart_cord, lengths, angles, num_atoms)
            
            
            a, f_base, mask_a_or_f, mask_f = self._to_dense(
                batch, atom_types=atom_types, frac_coords=fract_disp, constraints=constraints
            )
            flat_base = self._to_flat(a, f_base, lattices, dims)
        elif sample:
            
            num_atoms = self._get_num_atoms(mask_a_or_f)
            cart_cord = frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
            
            t = torch.tensor(0.5)  
            D_sample = self.sample_D(t, shape=cart_cord.shape)
            D_sample = torch.tensor(D_sample, device=cart_cord.device)

            mask_f = mask_f.float()                     # cast to float
            mask_f_unsq = mask_f.unsqueeze(1)           # reshape to [1025, 1]
            ones_vector = torch.ones(1, 3, dtype=torch.float32, device=cart_cord.device)
            m_f = mask_f_unsq.matmul(ones_vector)   
            print(m_f, "m_f") 
            
            D_sample = torch.where(m_f == 1, torch.zeros_like(D_sample), D_sample)
            cart_cord += D_sample
            fract_disp = cart_to_frac_coords(cart_cord, lengths, angles, num_atoms)
            
            a, f_base, mask_a_or_f, mask_f = self._to_dense(
                batch, atom_types=atom_types, frac_coords=fract_disp, constraints=constraints
            )
            
            # print(f_base.shape, "f_base shape")
            # print(f_base[0], "f_base")
            
            flat_base = self._to_flat(a, f_base, lattices, dims)
            
            
            
        flat = self._to_flat(a, f, lattices, dims)
        
        # if f_base is None:
        #     raise Exception("Velocitiies is None: f_base")
        
        # manifolds = [manifold.to(device=batch.device) for manifold in manifolds]
 
  
        # torch.Size([256, 61, 3])

        if split_manifold:
            return SplitManifoldGetterOut(
                flat,
                flat_base,
                *manifolds,
                dims,
                mask_a_or_f,
                mask_f
            )
        else:
            return ManifoldGetterOut(
                flat,
                flat_base,
                *manifolds,
                dims,
                mask_a_or_f,
                mask_f
            )

    def forward(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        split_manifold: bool,
        velocities: torch.Tensor = None,
        constraints: torch.Tensor = None,
        displace: bool = False,
        sample: bool = False
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """
        Main forward function.
        Converts loader data into georep and then to the flat manifold representation.
        """

        """converts data from the loader into the georep, then to the manifold flatrep"""
        # torch.Size([14896]) atom types shape
        # print(atom_types.shape, "atom types shape")
        
        # torch.Size([14896, 3]) frac_coords shape
        # print(frac_coords.shape, "frac_coords shape")
        
        # tensor([0.6160, 0.0584, 0.4866], device='cuda:0') frac_coords 0
        # print(frac_coords[0], "frac_coords 0")
        
        
        atom_types = self._convert_atom_types(atom_types)
        # torch.Size([14896, 100]) atom types shape
        # print(atom_types.shape, "atom types shape")
        
        # lattice params
        # print(self.lattice_manifold, "lattice type manifold") 
        

        if "spd" in self.lattice_manifold:
            lattices = lattice_params_to_spd_vector(lengths, angles)
        elif self.lattice_manifold == "non_symmetric":
            lattices = lattice_params_to_matrix_torch(lengths, angles)
        elif self.lattice_manifold == "lattice_params":
            lattices_deg = LatticeParams.cat(lengths, angles)
            lattices = LatticeParams().deg2uncontrained(lattices_deg)
        elif self.lattice_manifold == "lattice_params_normal_base":
            lattices = LatticeParams.cat(lengths, angles)
        else:
            raise NotImplementedError()
        
        # tensor([0.6160, 0.0584, 0.4866], device='cuda:0') frac_coords 0
        # print(frac_coords[0], "frac_coords 0")

        return self.georep_to_flatrep(
            batch, atom_types, frac_coords, lattices, split_manifold, velocities, constraints, lengths, angles, displace=displace, sample=sample
        )

    def _convert_atom_types(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Converts atom types into the required representation (one-hot, analog bits, etc.)
        based on the specified manifold.
        """
        if atom_types.ndim == 1:  # the types are NOT one_hot already
            if self.atom_type_manifold in ["simplex", "null_manifold"]:
                atom_types = self._atomic_one_hot(atom_types)  # B x NUM_ATOMIC_TYPES
            elif self.atom_type_manifold == "analog_bits":
                atom_types = self._atomic_bits(atom_types, self.analog_bits_scale)
            else:
                raise TypeError()
        return atom_types

    def from_empty_batch(
        self,
        batch: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """
        Handles the case where there are no atoms in the batch.
        Constructs a dummy manifold output based on the empty input.
        """
        _, mask_a_or_f = to_dense_batch(
            x=torch.zeros((*batch.shape, 1), device=batch.device), batch=batch
        )  # B x N
        num_atoms = self._get_num_atoms(mask_a_or_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms, None, dim_coords, split_manifold=split_manifold
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_a_or_f

    def from_only_atom_types(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """
        Similar to from_empty_batch, but starts from only atom types.
        Converts the provided atom types and builds the corresponding manifolds.
        """
        atom_types = self._convert_atom_types(atom_types)
        atom_types_dense_one_hot, mask_a_or_f = to_dense_batch(
            x=atom_types, batch=batch
        )  # B x N
        num_atoms = self._get_num_atoms(mask_a_or_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            atom_types_dense_one_hot,
            dim_coords,
            split_manifold=split_manifold,
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_a_or_f

    @staticmethod
    def _from_dense(
        a: torch.Tensor,
        f: torch.Tensor,
        mask_a_or_f: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts only the valid (non-padded) atom and coordinate data based on the mask.
        """
        return a[mask_a_or_f], f[mask_a_or_f]

    def _from_flat(
        self,
        flat: torch.Tensor,
        dims: Dims,
        max_num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the flat representation into its components:
        atom types, fractional coordinates, and lattice parameters.
        Reshapes the parts according to the manifold settings.
        """
        a, f, l, _ = torch.tensor_split(flat, np.cumsum(dims).tolist(), dim=1)

        if self.atom_type_manifold in ["simplex", "null_manifold"]:
            a = a.reshape(-1, max_num_atoms, NUM_ATOMIC_TYPES)
        elif self.atom_type_manifold == "analog_bits":
            a = a.reshape(-1, max_num_atoms, NUM_ATOMIC_BITS)
        else:
            raise TypeError()

        if "spd" in self.lattice_manifold:
            l = l.reshape(-1, dims.l)
        elif self.lattice_manifold == "non_symmetric":
            l = l.reshape(-1, int(math.sqrt(dims.l)), int(math.sqrt(dims.l)))
        elif (
            self.lattice_manifold == "lattice_params"
            or self.lattice_manifold == "lattice_params_normal_base"
        ):
            l = l.reshape(-1, dims.l)
        else:
            raise NotImplementedError()

        return a, f.reshape(-1, max_num_atoms, dims.f // max_num_atoms), l

    def flatrep_to_georep(
        self, flat: torch.Tensor, dims: Dims, mask_a_or_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts the flat manifold representation back into the georepresentation,
        consisting of atom types, fractional coordinates, and lattice parameters.
        """
        """converts from the manifold flatrep to the georep"""
        max_num_atoms = self._get_max_num_atoms(mask_a_or_f)
        a, f, l = self._from_flat(flat, dims, max_num_atoms)
        a, f = self._from_dense(a, f, mask_a_or_f)
        return GeomTuple(a, f, l)

    def georep_to_crystal(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts the georepresentation into a crystal representation.
        For lattices, it converts the representation based on the specified lattice manifold.
        """
        """converts from the georep to (one_hot / bits, frac_coords, lattice matrix)"""
        if "spd" in self.lattice_manifold:
            lattices = spd_vector_to_lattice_matrix(lattices)
        elif self.lattice_manifold == "non_symmetric":
            pass
        elif self.lattice_manifold == "lattice_params":
            lattices_deg = LatticeParams().uncontrained2deg(lattices)
            lengths, angles_deg = LatticeParams.split(lattices_deg)
            lattices = lattice_params_to_matrix_torch(lengths, angles_deg)
        elif self.lattice_manifold == "lattice_params_normal_base":
            lengths, angles_deg = LatticeParamsNormalBase.split(lattices)
            lattices = lattice_params_to_matrix_torch(lengths, angles_deg)
        else:
            raise NotImplementedError()

        return atom_types, frac_coords, lattices

    def flatrep_to_crystal(
        self, flat: torch.Tensor, dims: Dims, mask_a_or_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a flat manifold representation directly into the crystal representation.
        """
        """converts from the manifold flatrep to (one_hot / bits, frac_coords, lattice matrix)"""
        return self.georep_to_crystal(*self.flatrep_to_georep(flat, dims, mask_a_or_f))

    @staticmethod
    def mask_a_or_f_to_batch(mask_a_or_f: torch.BoolTensor) -> torch.LongTensor:
        """
        Converts a 2D mask to a batch index vector.
        """
        return mask_2d_to_batch(mask_a_or_f)

    @staticmethod
    def _get_manifold(
        num_atom: int,
        atom_types_dense_one_hot: torch.Tensor | None,
        dim_coords: int,
        max_num_atoms: int,
        batch_idx: int,
        split_manifold: bool,
        atom_type_manifold: atom_type_manifold_types,
        coord_manifold: coord_manifold_types,
        lattice_manifold: lattice_manifold_types,
        dataset: dataset_options | None = None,
        analog_bits_scale: float | None = None,
        length_inner_coef: float | None = None,
    ) -> (
        tuple[ProductManifoldWithLogProb]
        | tuple[
            ProductManifoldWithLogProb,
            MultiAtomAnalogBits
            | MultiAtomFlatDirichletSimplex
            | NullManifoldWithDeltaRandom,
            FlatTorus01FixFirstAtomToOrigin
            | FlatTorus01FixFirstAtomToOriginWrappedNormal
            | MaskedNoDriftFlatTorus01
            | MaskedNoDriftFlatTorus01WrappedNormal,
            EuclideanWithLogProb | SPDGivenN | LatticeParams,
        ]
    ):
        """
        Instantiates and returns the appropriate manifold objects for atom types,
        coordinates, and lattice parameters based on the provided settings.
        """

        if atom_type_manifold == "simplex":
            a_manifold = (
                MultiAtomFlatDirichletSimplex(
                    num_categories=NUM_ATOMIC_TYPES,
                    num_atoms=num_atom,
                    max_num_atoms=max_num_atoms,
                ),
                NUM_ATOMIC_TYPES * max_num_atoms,
            )
        elif atom_type_manifold == "null_manifold":
            a_manifold = (
                NullManifoldWithDeltaRandom(
                    atom_types_dense_one_hot[batch_idx].reshape(-1)
                ),
                NUM_ATOMIC_TYPES * max_num_atoms,
            )
        elif atom_type_manifold == "analog_bits":
            a_manifold = (
                MultiAtomAnalogBits(
                    analog_bits_scale,
                    num_bits=NUM_ATOMIC_BITS,
                    num_atoms=num_atom,
                    max_num_atoms=max_num_atoms,
                ),
                NUM_ATOMIC_BITS * max_num_atoms,
            )
        else:
            raise ValueError(
                f"{atom_type_manifold=} not in {atom_type_manifold_types=}"
            )

        if coord_manifold == "flat_torus_01":
            f_manifold = (
                MaskedNoDriftEuclidean(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_normal":
            f_manifold = (
                MaskedNoDriftEuclidean(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_fixfirst":
            f_manifold = (
                MaskedNoDriftEuclidean(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_fixfirst_normal":
            f_manifold = (
                MaskedNoDriftEuclidean(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        else:
            raise ValueError(f"{coord_manifold=} not in {coord_manifold_types=}")

        if lattice_manifold == "non_symmetric":
            l_manifold = (EuclideanWithLogProb(ndim=1), dim_coords**2)
        elif lattice_manifold == "spd_euclidean_geo":
            spd = SPDGivenN.from_dataset(
                num_atom,
                dataset,
                Riem_geodesic=False,
            )
            l_manifold = (spd, spd.vecdim(dim_coords))
        elif lattice_manifold == "spd_riemanian_geo":
            spd = SPDGivenN.from_dataset(
                num_atom,
                dataset,
                Riem_geodesic=True,
            )
            l_manifold = (spd, spd.vecdim(dim_coords))
        elif lattice_manifold == "lattice_params":
            lp = LatticeParams.from_dataset(dataset, length_inner_coef)
            l_manifold = (lp, LatticeParams.dim(dim_coords))
        elif lattice_manifold == "lattice_params_normal_base":
            lp = LatticeParamsNormalBase()
            l_manifold = (lp, LatticeParamsNormalBase.dim(dim_coords))
        else:
            raise ValueError(f"{lattice_manifold=} not in {lattice_manifold_types=}")

        manifolds = [a_manifold] + [f_manifold] + [l_manifold]
        if split_manifold:
            return (
                ProductManifoldWithLogProb(*manifolds),
                a_manifold[0],
                f_manifold[0],
                l_manifold[0],
            )
        else:
            return ProductManifoldWithLogProb(*manifolds)

    def get_dims(self, dim_coords: int, max_num_atoms: int) -> Dims:
        """
        Computes the dimension for the flat representation based on:
        - Atom type representation (one-hot or analog bits)
        - Fractional coordinates dimension
        - Lattice parameters representation
        """
        if self.atom_type_manifold in ["null_manifold", "simplex"]:
            dim_a = NUM_ATOMIC_TYPES * max_num_atoms
        elif self.atom_type_manifold == "analog_bits":
            dim_a = NUM_ATOMIC_BITS * max_num_atoms
        else:
            raise NotImplementedError("")

        dim_f = dim_coords * max_num_atoms

        if self.lattice_manifold == "non_symmetric":
            dim_l = dim_coords**2
        elif "spd" in self.lattice_manifold:
            dim_l = SPDGivenN.vecdim(dim_coords)
        elif (
            self.lattice_manifold == "lattice_params"
            or self.lattice_manifold == "lattice_params_normal_base"
        ):
            dim_l = LatticeParams.dim(dim_coords)
        else:
            raise NotImplementedError("")

        return Dims(dim_a, dim_f, dim_l)

    def get_manifolds(
        self,
        num_atoms: torch.LongTensor,
        atom_types_dense_one_hot: torch.Tensor | None,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
       
        tuple[VMapManifolds, tuple[int, int, int]]
        | tuple[
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            tuple[int, int, int],
        ]
    ):
        """
        Creates a VMapManifolds object for each batch element by iterating over the number
        of atoms in each graph.
        """
        max_num_atoms = num_atoms.amax(0).cpu().item()

        out_manifolds, a_manifolds, f_manifolds, l_manifolds = [], [], [], []
        for batch_idx, num_atom in enumerate(num_atoms):
            manis = self._get_manifold(
                num_atom,
                atom_types_dense_one_hot,
                dim_coords,
                max_num_atoms,
                batch_idx,
                split_manifold=split_manifold,
                atom_type_manifold=self.atom_type_manifold,
                coord_manifold=self.coord_manifold,
                lattice_manifold=self.lattice_manifold,
                dataset=self.dataset,
                analog_bits_scale=self.analog_bits_scale,
                length_inner_coef=self.length_inner_coef,
            )
            if split_manifold:
                out_manifolds.append(manis[0])
                a_manifolds.append(manis[1])
                f_manifolds.append(manis[2])
                l_manifolds.append(manis[3])
            else:
                out_manifolds.append(manis)

        if split_manifold:
            return (
                VMapManifolds(out_manifolds),
                VMapManifolds(a_manifolds),
                VMapManifolds(f_manifolds),
                VMapManifolds(l_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
        else:
            return (
                VMapManifolds(out_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
