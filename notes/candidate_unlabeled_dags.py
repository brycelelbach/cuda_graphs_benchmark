#! /usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
# Reply-To: brycelelbach@gmail.com
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
###############################################################################

from sys import stdout

from itertools import product

from argparse import ArgumentParser as argument_parser

from csv import writer as csv_writer

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

n   = args.size
T_n = triangular_number(n - 1) # `((n - 1) * n)) / 2`

print "The number of strictly upper (or lower) triangular square matrices " + \
      "of size " + str(n) + " is " + str(2 ** T_n)
print

matrices = []

unique = []

# Sorted Rows -> Index
row_matcher = {}

# Sorted Cols -> Index
col_matcher = {}

# Iterate over the set of `T_n` element tuples formed be selecting (with
# repetition) from the 2-element alphabet `[0, 1]`.
for t in map(lambda x: list(x), product(range(2), repeat = T_n)):
  next_index = len(matrices)

  rows = []
  cols = []

  for i in range(n):
    rows.append([0] * n)
    cols.append([0] * n)

  for i in range(n):
    for j in range(n):
      if j > i:
        # Consume the next element of the current tuple.
        v = t.pop(0)
        rows[i][j] = v
        cols[j][i] = v

  #for i in range(n):
  #  print rows[i], cols[i]
  #print

  # Convert `rows` and `cols` to `tuple` types so that they're hashable.
  rows = tuple(tuple(x) for x in rows)
  cols = tuple(tuple(x) for x in cols)

  matrices.append(rows)

  sorted_rows = tuple(sorted(rows))
  sorted_cols = tuple(sorted(cols))

  u = (True, None)

  if sorted_rows in row_matcher:
    # We aren't unique, we're isomorphic to some other graph.
    row_matcher[sorted_rows].append(next_index)
    u = [False, row_matcher[sorted_rows][0]]
  else: 
    # We might be unique.
    row_matcher[sorted_rows] = [next_index]

  if sorted_cols in col_matcher:
    # We aren't unique, we're isomorphic to some other graph.
    col_matcher[sorted_cols].append(next_index)
    if u[0]:
      u = [False, col_matcher[sorted_cols][0]]
    else:
      u.append(col_matcher[sorted_cols][0])
  else: 
    # We might be unique.
    col_matcher[sorted_cols] = [next_index]

  unique.append(u)

unlabeled = filter(lambda x: unique[x][0], range(len(matrices)))

print "The number of DAGs with " + str(n) + " labeled vertices is " + \
      str(len(unlabeled))
print

matrix_writer = csv_writer(stdout, lineterminator = '\n')

for i in range(len(matrices)):
  #print i, unique[i]
  if unique[i][0]:
    for row in matrices[i]:
      matrix_writer.writerow(row)
    print

