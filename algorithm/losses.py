import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardCorrectedCELoss(nn.Module):
    """Forward loss correction: softmax logits -> multiply by T^T in probability space.
    We implement by transforming predicted class probabilities p into 

    q = p @ T, then compute CE with noisy label (as in 'forward' correction).
    Reference idea: Patrini et al., 2017.
    """
    def __init__(self, T):
        super().__init__()
        self.register_buffer('T', T)  # shape (C, C)

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)  # (B, C)
        q = torch.clamp(p @ self.T, 1e-12, 1.0)  # (B, C)
        log_q = torch.log(q)
        loss = F.nll_loss(log_q, targets, reduction='mean')
        return loss

class GCELoss(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018).
    For q in (0,1], loss = (1 - p_y^q) / q. When q->0, approaches CE; when q=1, MAE.
    """
    def __init__(self, q=0.7):
        super().__init__()
        assert 0 < q <= 1.0
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)  # (B, C)
        py = p.gather(1, targets.view(-1,1)).clamp(1e-12, 1.0)
        loss = (1.0 - py.pow(self.q)) / self.q
        return loss.mean()
