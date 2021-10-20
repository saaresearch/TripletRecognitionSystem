import torch.nn as nn
import torch

from typing import Tuple

class TripletMiner( nn.Module ):
    """Triplet Miner
    Tripelt Mining base class.
    Attributes
    ----------
    margin: int
            Margin distance between positive and negative samples from anchor
            perspective. Default to 0.5.
    """
    def __init__( self: 'TripletMiner', margin: int = .5 ) -> None:
        """Init
        Parameters
        ----------
        margin: int
                Margin distance between positive and negative samples from
                anchor perspective. Default to 0.5.
        """
        super( TripletMiner, self ).__init__( )
        self.margin = margin

    def _pairwise_dist(
        self      : 'TripletMiner',
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """PairWiseDist
        Parameters
        ----------
        embeddings: torch.Tensor
                    Embeddings is the ouput of the neural network given data
                    samples from the dataset.
        Returns
        -------
        pairwise_dist: torch.Tensor
                       Pairwise distances between each samples.
        """
        dot_product          = torch.matmul( embeddings, embeddings.t( ) )
        square_norm          = torch.diag( dot_product )
        pairwise_dist        = square_norm.unsqueeze( 0 ) - \
                               2. * dot_product + square_norm.unsqueeze( 1 )
        pdn                  = pairwise_dist < 0.
        pairwise_dist[ pdn ] = 0.
        return pairwise_dist

    def forward(
        self      : 'TripletMiner',
        ids       : torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[ torch.Tensor ]:
        """Forward
        Parameters
        ----------
        ids       : torch.Tensor
                    Labels of samples from the dataset respectively to the
                    embeddings.
        embeddings: torch.Tensor
                    Embeddings is the ouput of the neural network given data
                    samples from the dataset.
        Raises
        ------
        NotImplementedError: Based class does not provide implementation of the
                             mining technique.
        """
        raise NotImplementedError( 'Mining function not implemented yet!' )

class AllTripletMiner( TripletMiner ):
    """AllTripletMiner
    The class provides mining for all valid triplet from a given dataset.
    Attributes
    ----------
    margin: int
            Margin distance between positive and negative samples from anchor
            perspective. Default to 0.5.
    """

    def __init__( self: 'AllTripletMiner', margin: int = .5 ) -> None:
        """Init
        Params
        ------
        margin: int
                Margin distance between positive and negative samples from anchor
                perspective. Default to 0.5.
        """
        super( AllTripletMiner, self ).__init__( margin )

    def _triplet_mask( self: 'AllTripletMiner', ids: torch.Tensor ) -> torch.Tensor:
        """TripletMask
        Parameters
        ----------
        ids: torch.Tensor
             Labels of samples from the dataset respectively to the
             embeddings.
        Returns
        -------
        mask: torch.Tensor
              Mask for every valid triplet from the selected samples.
        """
        eye          = torch.eye( ids.size( 0 ), requires_grad = False ).cuda( ) if ids.is_cuda else \
                       torch.eye( ids.size( 0 ), requires_grad = False  )

        ids_not_eq   = ( 1 - eye ).bool( )
        i_not_eq_j   = ids_not_eq.unsqueeze( 2 )
        i_not_eq_k   = ids_not_eq.unsqueeze( 1 )
        j_not_eq_k   = ids_not_eq.unsqueeze( 0 )
        distinct_idx = ( ( i_not_eq_j & i_not_eq_k ) & j_not_eq_k )

        ids_eq       = ids.unsqueeze( 0 ) == ids.unsqueeze( 1 )
        i_eq_j       = ids_eq.unsqueeze( 2 )
        i_eq_k       = ids_eq.unsqueeze( 1 )

        valid_ids    = ( i_eq_j & ~i_eq_k )
        mask         = distinct_idx & valid_ids
        return mask

    def forward(
        self      : 'AllTripletMiner',
        ids       : torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[ torch.Tensor ]:
        """Forward
        Parameters
        ----------
        ids       : torch.Tensor
                    Labels of samples from the dataset respectively to the
                    embeddings.
        embeddings: torch.Tensor
                    Embeddings is the ouput of the neural network given data
                    samples from the dataset.
        Returns
        -------
        loss     : torch.Tensor
                   Loss obtained with the AllTripletMiner sampling technique.
        f_pos_tri: torch.Tensor
                   Proportion of postive triplets. Less is better. The value
                   should decrease with training.
        """
        pairwise_dist = self._pairwise_dist( embeddings )

        pos_dist      = pairwise_dist.unsqueeze( 2 )
        neg_dist      = pairwise_dist.unsqueeze( 1 )

        mask          = self._triplet_mask( ids ).float( )
        loss          = pos_dist - neg_dist + self.margin
        loss         *= mask
        loss          = torch.clamp( loss, min = 0. )

        n_pos_tri     = torch.sum( ( loss > 1e-16 ).float( ) )
        n_val_tri     = torch.sum( mask )
        f_pos_tri     = n_pos_tri / ( n_val_tri + 1e-16 )

        loss          = torch.sum( loss ) / ( n_pos_tri + 1e-16 )

        return loss, f_pos_tri

class HardNegativeTripletMiner( TripletMiner ):
    """HardNegativeTripletMiner
    The class provides mining for hard negative triplet only.
    Attributes
    ----------
    margin: int
            Margin distance between positive and negative samples from anchor
            perspective. Default to 0.5.
    """

    def __init__( self: 'HardNegativeTripletMiner', margin: int = .5 ) -> None:
        """Init
        Params
        ------
        margin: int
                Margin distance between positive and negative samples from anchor
                perspective. Default to 0.5.
        """
        super( HardNegativeTripletMiner, self ).__init__( margin )

    def _pos_dist(
        self         : 'HardNegativeTripletMiner',
        ids          : torch.Tensor,
        pairwise_dist: torch.Tensor
    ) -> torch.Tensor:
        """PositiveDistances
        Parameters
        ----------
        ids          : torch.Tensor
                       Labels of samples from the dataset respectively to the
                       embeddings.
        pairwise_dist: torch.Tensor
                       Pairwise distances between each samples.
        Returns
        -------
        anc_pos_dist: torch.Tensor
                      Distances between positives and anchors.
        """
        eye              = torch.eye( ids.size( 0 ), requires_grad = False ).cuda( ) if ids.is_cuda else \
                           torch.eye( ids.size( 0 ), requires_grad = False )

        mask_anc_pos     = ( ~eye.bool( ) & ( ids.unsqueeze( 0 ) == ids.unsqueeze( 1 ) ) )
        anc_pos_dist     = mask_anc_pos.float( ) * pairwise_dist
        anc_pos_dist, _  = anc_pos_dist.max( axis = 1, keepdim = True )
        return anc_pos_dist

    def _neg_dist(
        self      : 'HardNegativeTripletMiner',
        ids          : torch.Tensor,
        pairwise_dist: torch.Tensor
    ) -> torch.Tensor:
        """NegativeDistances
        Parameters
        ----------
        ids          : torch.Tensor
                       Labels of samples from the dataset respectively to the
                       embeddings.
        pairwise_dist: torch.Tensor
                       Pairwise distances between each samples.
        Returns
        -------
        anc_neg_dist: torch.Tensor
                      Distances between negatives and anchors.
        """
        mask_anc_neg        = ids.unsqueeze( 0 ) != ids.unsqueeze( 1 )
        max_anc_neg_dist, _ = pairwise_dist.max( axis = 1, keepdim = True )
        anc_neg_dist        = pairwise_dist + \
                              max_anc_neg_dist * ( 1. - mask_anc_neg.float( ) )
        anc_neg_dist        = anc_neg_dist.mean( axis = 1, keepdim = False )
        return anc_neg_dist

    def forward(
        self      : 'HardNegativeTripletMiner',
        ids       : torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[ torch.Tensor ]:
        """Forward
        Parameters
        ----------
        ids       : torch.Tensor
                    Labels of samples from the dataset respectively to the
                    embeddings.
        embeddings: torch.Tensor
                    Embeddings is the ouput of the neural network given data
                    samples from the dataset.
        Returns
        -------
        loss     : torch.Tensor
                   Loss obtained with the HardNegativeTripletMiner sampling
                   technique.
        _        : None
                   To match the format of the AllTripletMiner
        """
        pairwise_dist = self._pairwise_dist( embeddings )
        pos_dist      = self._pos_dist( ids, pairwise_dist )
        neg_dist      = self._neg_dist( ids, pairwise_dist )

        loss          = torch.clamp( pos_dist - neg_dist + self.margin, min = 0. )
        loss          = loss.mean( )
        return loss, None

import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Callable
from typing import Tuple

class TripletDataset( Dataset ):
    """TripletDataset
    The TripletDataset extends the standard Dataset provided by the pytorch
    utils. It provides simple access to data with the possibility of returning
    more than one sample per index based on the label.
    Attributes
    ----------
    labels  : np.ndarray
              Array containing all the labels respectively to each data sample.
              Labels needs to provide a way to access a sample label by index.
    data_fn : Callable
              The data_fn provides access to sample data given its index in the
              dataset. Providding a function instead of array has been chosen
              for preprocessing and other reasons.
    size    : int
              Size gives the dataset size, number of samples.
    n_sample: int
              The value represents the number of sample per index. The other
              samples will be chosen to be the same as the selected one. This
              allows to augment the number of possible valid triplet when used
              with a tripelt mining strategy.
    """

    def __init__(
        self    : 'TripletDataset',
        labels  : np.ndarray,
        data_fn : Callable,
        size    : int,
        n_sample: int
    ) -> None:
        """Init
        Parameters
        ----------
        labels  : np.ndarray
                  Array containing all the labels respectively to each data
                  sample. Labels needs to provide a way to access a sample label
                  by index.
        data_fn : Callable
                  The data_fn provides access to sample data given its index in
                  the dataset. Providding a function instead of array has been
                  chosen for preprocessing and other reasons.
        size    : int
                  Size gives the dataset size, number of samples.
        n_sample: int
                  The value represents the number of sample per index. The other
                  samples will be chosen to be the same as the selected one.
                  This allows to augment the number of possible valid triplet
                  when used with a tripelt mining strategy.
        """
        super( Dataset, self ).__init__( )
        self.labels   = labels
        self.data_fn  = data_fn
        self.size     = size
        self.n_sample = n_sample

    def __len__( self: 'TripletDataset' ) -> int:
        """Len
        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __getitem__( self: 'TripletDataset', index: int ) -> Tuple[ np.ndarray ]:
        """GetItem
        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.
        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        data  : torch.Tensor
                Returns the data respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """
        if not ( index >= 0 and index < len( self ) ):
            raise IndexError( f'Index { index } is out of range [ 0, { len( self ) } ]' )

        label         = np.array( [  self.labels[ index ] ] )
        datum         = np.array( [ self.data_fn( index ) ] )

        if self.n_sample == 1:
            return label, datum

        mask          = self.labels == label
        mask[ index ] = False
        mask          = mask.astype( np.float32 )

        indexes       = mask.nonzero( )[ 0 ]
        indexes       = np.random.choice( indexes, self.n_sample - 1, False )
        data          = np.array( [  self.data_fn( i ) for i in indexes ] )

        labels        = np.repeat( label, self.n_sample )
        data          = np.concatenate( ( datum, data ), axis = 0 )

        labels        = torch.from_numpy( labels )
        data          = torch.from_numpy(   data )

        return labels, data