import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.007, ensemble = True, num_ensembles = 10):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.ensemble = ensemble
        self.num_ensembles = num_ensembles

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        
        if self.ensemble:
            labels = labels.unsqueeze(1).expand(*labels.shape[:1], self.num_ensembles, *labels.shape[1:])
        
        if self._targeted:
            target_labels = self._get_target_label(images, labels)
            
            

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        #Amin Edits
        if not self.ensemble:
            outputs = self.model(images)
        else:
            outputs, _ = self.model(images)
            
            if outputs.ndim == 3:
                outputs = outputs.reshape(outputs.shape[0] * labels.shape[1], -1)
            if labels.ndim > 1:
                labels = labels.reshape(-1)
        
        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
