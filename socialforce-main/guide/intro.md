# Introduction

[![Tests](https://github.com/svenkreiss/socialforce/actions/workflows/tests.yml/badge.svg)](https://github.com/svenkreiss/socialforce/actions/workflows/tests.yml)<br />
[GitHub repository](https://github.com/svenkreiss/socialforce).<br />
[Deep Social Force (arXiv:2109.12081)](https://arxiv.org/abs/2109.12081).

Install with:

```
pip install socialforce
```


# Abstract

> [__Deep Social Force__](https://arxiv.org/abs/2109.12081)<br />
> _[Sven Kreiss](https://www.svenkreiss.com)_, 2021.
>
> The Social Force model introduced by Helbing and Molnar in 1995
> is a cornerstone of pedestrian simulation. This paper
> introduces a differentiable simulation of the Social Force model
> where the assumptions on the shapes of interaction potentials are relaxed
> with the use of universal function approximators in the form of neural
> networks.
> Classical force-based pedestrian simulations suffer from unnatural
> locking behavior on head-on collision paths. In addition, they cannot
> model the bias
> of pedestrians to avoid each other on the right or left depending on
> the geographic region.
> My experiments with more general interaction potentials show that
> potentials with a sharp tip in the front avoid
> locking. In addition, asymmetric interaction potentials lead to a left or right
> bias when pedestrians avoid each other.


## Acknowledgement

Funded by the SNSF under Spark grant 190677.
