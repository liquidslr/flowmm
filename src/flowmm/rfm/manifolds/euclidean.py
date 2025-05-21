"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
from geoopt import Euclidean
from geoopt.utils import size2shape
from flowmm.rfm.manifolds.masked import MaskedManifold
from typing import Union, Tuple

class EuclideanWithLogProb(Euclidean):
    def random_normal(
        self, *size, mean=0.0, std=1.0, device=None, dtype=None
    ) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
        return torch.randn(*size, device=mean.device, dtype=mean.dtype) * std + mean

    def random_base(self, *args, **kwargs):
        return self.random_normal(*args, **kwargs)

    random = random_base

    def normal_logprob(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.ndim == 0:
            raise NotImplementedError()
        else:
            dist = torch.distributions.normal.Normal(
                torch.zeros_like(x[-self.ndim :]),
                torch.ones_like(x[-self.ndim :]),
                validate_args=False,  # doesn't work with vmap yet https://github.com/pytorch/functorch/issues/257
            )
            dist = torch.distributions.independent.Independent(dist, 1)
        return dist.log_prob(x)

    def base_logprob(self, *args, **kwargs) -> torch.Tensor:
        return self.normal_logprob(*args, **kwargs)

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)


class EuclideanManifold(Euclidean):
    """Represents a Euclidean manifold.

    Implements standard operations in Euclidean space.
    """
    name = "EuclideanManifold"
    reversible = True  # Typically reversible in Euclidean space

    def __init__(self, ndim=1):
        super().__init__(ndim=ndim)

    @staticmethod
    def expmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (x + u) % 1.0

    @staticmethod
    def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y - x) 

    @staticmethod
    def projx(x: torch.Tensor) -> torch.Tensor:
        return x % 1.0

    @staticmethod
    def random_uniform(*size, dtype=None, device=None) -> torch.Tensor:
        # Sample uniformly from [0, 1) in Euclidean space.
        return torch.rand(*size, dtype=dtype, device=device)

    random = random_uniform
    
    @staticmethod
    def uniform_logprob(x: torch.Tensor) -> torch.Tensor:
        #  return -0.5 * torch.sum(x**2, dim=-1) 
        return torch.full_like(x[..., 0], 0.0)

    def extra_repr(self):
        return "ndim={}".format(self.ndim)



class MaskedNoDriftEuclidean(MaskedManifold, EuclideanManifold):
    """Represents a Euclidean manifold with a masked subspace.

    In this configuration, only a subset of the atoms (e.g. atoms 2 to N) are free
    to move while the first atom is fixed to remove overall translation (drift).
    
    The flat representation is defined on a subspace of ℝ^(max_num_atoms*dim_coords),
    but only the first `num_atoms` entries (as specified by a mask) are considered active.
    """
    name = "MaskedNoDriftEuclidean"
    reversible = True

    def __init__(self, dim_coords: int, num_atoms: int, max_num_atoms: int):
        super().__init__(ndim=1)
        self.dim_coords = dim_coords
        # Create a boolean mask of shape [max_num_atoms, 1]
        # where the first `num_atoms` entries are True (active) and the rest are False.
        mask = torch.zeros(max_num_atoms, dtype=torch.bool)
        mask[:num_atoms] = torch.ones(num_atoms, dtype=torch.bool)
        self.register_buffer("mask", mask.unsqueeze(-1))

    @property
    def dim_m1(self) -> int:
        return self.dim_coords

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False) -> torch.Tensor:
        if v is None:
            v = u
        return torch.sum(u * v, dim=-1, keepdim=keepdim)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Projects the tangent vector u onto the tangent space by removing drift."""
        # initial_shape = u.shape
        mean = torch.mean(u, dim=-1, keepdim=True)  # Mean over all atoms
        return u - mean 

    def extra_repr(self):
        return f"dim_coords={self.dim_coords}"
