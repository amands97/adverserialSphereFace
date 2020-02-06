import torch


def gumbel_softmax(logits, tau=1, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    # - log (u_i) is effectively a sample from exponential distribution
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    # if hard:
        # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    # print(index.shape)
    
    print(index)
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    # print(ret[])
    print(ret)
    print(ret[:, 0])
    return ret[:, 0]

def gumbel_bernoulli(logits, tau=1, dim = -1):
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log() # ~Gumbel(0,1)
    gumbels = (logits + gumbels)/tau  # ~Gumbel(logits,tau)
    # y_soft = gu
    print(gumbels)
    y_soft = gumbels.softmax(dim)
    print(y_soft)
    index = y_soft.max(dim, keepdim=True)[1]
    # stack with zeros

    print(index)

logits = torch.randn(3)
logits2 = torch.stack((logits, 1- logits), -1)

print(logits2)
gumbel_softmax(logits2)
# gumbel_bernoulli(logits2)