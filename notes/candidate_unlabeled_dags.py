#! /usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
# Reply-To: brycelelbach@gmail.com
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
###############################################################################

from argparse import ArgumentParser as argument_parser

###############################################################################

def triangular_number(n):
  """Returns the `n`th partial sum of the series of positive integers, also
  known as the `n`th triangular number.

  .. math::

    T_n = \sum_{k=1}^{n} k = \frac{n(n+1)}{2}

    T_1 = 1
    T_2 = 3
    T_3 = 6
    T_4 = 10
    T_5 = 15
    T_6 = 21
 
  Args:
    n (int) : The upper bound of the summation.
  """

  return (n * (n + 1)) / 2

###############################################################################

def process_program_arguments():
  ap = argument_parser(
    description = (
      "Generates the set of all `(0, 1)` strictly upper triangular square "
      "matrices of size `n`."
    )
  )

  ap.add_argument(
    "size",
    help = ("The dimension of the square matrix."),
    type = int, default = 4,
    metavar = "N"
  )

  return ap.parse_args()

###############################################################################

args = process_program_arguments()

print "The number of strictly upper (or lower) triangular square matrices of " + \
      "size " + str(args.size) + " is " + str(2 ** triangular_number(args.size - 1))

