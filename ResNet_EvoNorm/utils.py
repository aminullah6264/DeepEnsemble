import torch
import numpy as np
import torch.nn.functional as F



def classification_loss(output, target):
    """Computes the cross entropy loss with respect to a batch of predictions and
    targets.
    
    Args:
        output (Tensor): predictions of shape ``[B, D]`` or ``[B, N, D]``.
        target (Tensor): targets of shape ``[B]``, ``[B, 1]``, ``[B, N]``,
            or ``[B, N, 1]``.

    Returns:
        LossInfo containing the computed cross entropy loss and the average
            accuracy.
    """

    # import ipdb; ipdb.set_trace()


    if output.ndim == 2:
        output = output.reshape(output.shape[0], target.shape[1], -1)
    pred = output.max(-1)[1]
    target = target.squeeze(-1)
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    if output.ndim == 3:
        output = output.transpose(1, 2)
    else:
        output = output.reshape(output.shape[0] * target.shape[1], -1)
        target = target.reshape(-1)
    loss = F.cross_entropy(output, target, reduction='sum')
    return loss, avg_acc


def _classification_vote(output, target, _voting = 'soft'):
        """Ensemble the ooutputs from sampled classifiers."""
        num_particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if _voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif _voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = []
            for i in range(pred.shape[0]):
                values, counts = torch.unique(
                    pred[i], sorted=False, return_counts=True)
                modes = (counts == counts.max()).nonzero()
                label = values[torch.randint(len(modes), (1, ))]
                vote.append(label)
            vote = torch.as_tensor(vote, device='cpu')
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        # loss = classification_loss(output, target)
        return [], correct