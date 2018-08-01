#! /usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
# Reply-To: brycelelbach@gmail.com
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
###############################################################################

from math import factorial

from argparse import ArgumentParser as argument_parser

###############################################################################

def range1idx(n):
  """Equivalent to: `range(1, n + 1)`"""
  return range(1, n + 1)

def binomial(x, y):
  """Returns the binomial coefficient of `n` and `k`:

  .. math::

    \binom{n}{k} = \frac{n!}{k!(n-k)!} \text{ for } 0 \leq k \leq n

  The result is `0` if `k` is less than `n` or either `k` is negative.

  Args:
    n (int) : Number of objects to choose from.
    k (int) : Number of objects to choose.
  """
  try:
    binom = factorial(x) // factorial(y) // factorial(x - y)
  except ValueError:
    binom = 0
  return binom

###############################################################################

def count_labeled_dags(n):
  """Returns the number of directed acyclic graphs with `n` labeled vertices.
  This is given by `OEIS Sequence A003024`_ and stated concise in
  `Kuipers 2013 Section 4`_:

  .. math::

    a_n = \sum_{k=1}^{n} (-1)^{k-1} \binom{n}{k} 2^{k(n-k)} a_{n-k} 

    a_1 = 1
    a_2 = 3
    a_3 = 25
    a_4 = 543
    a_5 = 29281
    a_6 = 3781503
 
  Args:
    n (int) : The number of vertices.

  .. _OEIS Sequence A003024:
    https://oeis.org/A003024

  .. _Kuipers 2013 Section 4:
    https://arxiv.org/pdf/1202.6590.pdf
  """

  if 1 >= n:
    return 1

  def iteration(k):
    return (-1) ** (k - 1) * binomial(n, k) * 2 ** (k * (n - k)) \
         * count_labeled_dags(n - k)
 
  return sum(map(iteration, range1idx(n)))
 
###############################################################################

def process_program_arguments():
  ap = argument_parser(
    description = (
      "Counts the number of acyclic directe graphs with `n` labeled vertices."
    )
  )

  ap.add_argument(
    "vertices",
    help = ("The number of vertices."),
    type = int, nargs = "*", default = range1idx(6),
    metavar = "N"
  )

  return ap.parse_args()

###############################################################################

args = process_program_arguments()

print "Number of DAGs with `n` labeled vertices:"

for n in args.vertices:
  print "a_" + str(n) + " = " + str(count_labeled_dags(n))

