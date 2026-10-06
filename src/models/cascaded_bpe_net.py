"""
Cascaded Deep Neural Network Architecture for Blood Pressure Estimation.
Replicating the multi-stage neural network architecture from:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).

Stage 1: MorphologyDNN
    - Input: 7 mPTP morphology features
    - Architecture: 7 -> 70 -> 100 -> 150 -> 2
    - Activation: Sigmoid for hidden layers, Linear for output
    - Output: Preliminary SBP and DBP estimates

Stage 2: SBPNet & DBPNet
    - SBPNet:
        - Input: 8 features (7 PRV dynamics features + preliminary SBP)
        - Architecture: 8 -> 10 -> 1
        - Activation: Sigmoid for hidden layer, Linear for output
        - Output: Final SBP
    - DBPNet:
        - Input: 8 features (7 PRV dynamics features + preliminary DBP)
        - Architecture: 8 -> 10 -> 1
        - Activation: Sigmoid for hidden layer, Linear for output
        - Output: Final DBP
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class MorphologyDNN(nn.Module):
    """
    Stage 1: Morphology-based preliminary BP estimation network.

    Takes 7 mPTP pulse morphology features and outputs preliminary
    estimates for both Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP).

    Architecture:
        Input (7) -> Linear(70) + Sigmoid -> Linear(100) + Sigmoid -> Linear(150) + Sigmoid -> Linear(2)
    """

    def __init__(self, in_features: int = 7, out_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 70),
            nn.Sigmoid(),
            nn.Linear(70, 100),
            nn.Sigmoid(),
            nn.Linear(100, 150),
            nn.Sigmoid(),
            nn.Linear(150, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of morphology features of shape (batch_size, 7).

        Returns
        -------
        torch.Tensor
            Preliminary [SBP, DBP] estimates of shape (batch_size, 2).
        """
        return self.net(x)


class SBPNet(nn.Module):
    """
    Stage 2: SBP refinement network.

    Takes 7 PRV dynamics features concatenated with the preliminary SBP estimate
    from Stage 1 (total 8 inputs) and outputs the final calibrated SBP estimate.

    Architecture:
        Input (8) -> Linear(10) + Sigmoid -> Linear(1)
    """

    def __init__(self, in_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated features of shape (batch_size, 8)
            [7 PRV dynamics features + preliminary SBP].

        Returns
        -------
        torch.Tensor
            Final SBP estimate of shape (batch_size, 1).
        """
        return self.net(x)


class DBPNet(nn.Module):
    """
    Stage 2: DBP refinement network.

    Takes 7 PRV dynamics features concatenated with the preliminary DBP estimate
    from Stage 1 (total 8 inputs) and outputs the final calibrated DBP estimate.

    Architecture:
        Input (8) -> Linear(10) + Sigmoid -> Linear(1)
    """

    def __init__(self, in_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated features of shape (batch_size, 8)
            [7 PRV dynamics features + preliminary DBP].

        Returns
        -------
        torch.Tensor
            Final DBP estimate of shape (batch_size, 1).
        """
        return self.net(x)


class CascadedBPENet(nn.Module):
    """
    Unified Cascaded Blood Pressure Estimation Network.

    Combines Stage 1 (MorphologyDNN) and Stage 2 (SBPNet, DBPNet) into an
    end-to-end multi-stage pipeline as described in the paper.
    """

    def __init__(
        self,
        morphology_dim: int = 7,
        dynamics_dim: int = 7,
    ):
        super().__init__()
        self.stage1_morphology = MorphologyDNN(in_features=morphology_dim, out_features=2)
        self.stage2_sbp = SBPNet(in_features=dynamics_dim + 1)
        self.stage2_dbp = DBPNet(in_features=dynamics_dim + 1)

    def forward(
        self,
        morphology_features: torch.Tensor,
        dynamics_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass through both stages.

        Parameters
        ----------
        morphology_features : torch.Tensor
            Tensor of shape (batch_size, 7) containing mPTP morphology features.
        dynamics_features : torch.Tensor
            Tensor of shape (batch_size, 7) containing PRV dynamics features.

        Returns
        -------
        final_sbp : torch.Tensor
            Final SBP estimate of shape (batch_size, 1).
        final_dbp : torch.Tensor
            Final DBP estimate of shape (batch_size, 1).
        prelim_sbp : torch.Tensor
            Stage 1 preliminary SBP estimate of shape (batch_size, 1).
        prelim_dbp : torch.Tensor
            Stage 1 preliminary DBP estimate of shape (batch_size, 1).
        """
        # Stage 1: Preliminary BP estimation from pulse morphology
        prelim_bp = self.stage1_morphology(morphology_features)  # (batch_size, 2)
        prelim_sbp = prelim_bp[:, 0:1]
        prelim_dbp = prelim_bp[:, 1:2]

        # Stage 2: Concatenate dynamics with preliminary estimates
        sbp_inputs = torch.cat([dynamics_features, prelim_sbp], dim=-1)  # (batch_size, 8)
        dbp_inputs = torch.cat([dynamics_features, prelim_dbp], dim=-1)  # (batch_size, 8)

        # Stage 2: Final calibrated SBP and DBP outputs
        final_sbp = self.stage2_sbp(sbp_inputs)  # (batch_size, 1)
        final_dbp = self.stage2_dbp(dbp_inputs)  # (batch_size, 1)

        return final_sbp, final_dbp, prelim_sbp, prelim_dbp
