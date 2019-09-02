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

from numpy import linalg

from numpy import array

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
      "Generates the set of all Laplacian matrices for directed acyclic graphs"
      "with `n` vertices."
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
eigenvalues = []

unique = []

# Sorted Rows -> {Eigenvalues : Index, Eigenvalues : Index}
# Each list of indices are isomorphic.
row_matcher = {}

# Sorted Cols -> {Eigenvalues : Index, Eigenvalues : Index}
col_matcher = {}

# Iterate over the set of `T_n` element tuples formed be selecting (with
# repetition) from the 2-element alphabet `[0, 1]`.
for t in map(lambda x: list(x), product(range(2), repeat = T_n)):
  next_index = len(matrices)

  # Initialize `rows` and `cols` matrices; they both store the same matrix, but
  # `rows` is row-wise and `cols` is column-wise.
  rows = []
  cols = []
  for i in range(n):
    rows.append([0] * n)
    cols.append([0] * n)

  # Build the upper-triangular adjacency matrix.
  for i in range(n):
    for j in range(n):
      if j > i:
        # Consume the next element of the current tuple.
        v = t.pop(0)
        rows[i][j] = v
        cols[j][i] = v

  # Initialize degree matrix.
  laplacian = []
  for i in range(n):
    laplacian.append([0] * n)

  # Generate the Laplacian matrix from the adjacency matrix.
  for i in range(n):
    for j in range(n):
      if j == i:
        laplacian[i][j] = sum(rows[i])
      if j > i:
        laplacian[i][j] = -rows[i][j]

  # Compute the eigvenvalues of the Laplacian.
  evs = tuple(linalg.eig(array(laplacian))[0])

  # Convert `rows` and `cols` to `tuple` types so that they're hashable.
  rows = tuple(tuple(x) for x in rows)
  cols = tuple(tuple(x) for x in cols)

  matrices.append(rows)
  eigenvalues.append(evs)

  sorted_rows = tuple(sorted(rows))
  sorted_cols = tuple(sorted(cols))

  u = (True, None)

  if sorted_rows in row_matcher:
    if evs[0] in row_matcher[sorted_rows]:
      # We aren't unique, we're isomorphic to some other graph.
      row_matcher[sorted_rows][evs].append(next_index)
      u = [False, row_matcher[sorted_rows][evs]]
    else:
      # We might be unique.
      row_matcher[sorted_rows][evs] = [next_index]
  else:
    # We might be unique.
    row_matcher[sorted_rows] = {evs : [next_index]}

  if sorted_cols in col_matcher:
    if evs in col_matcher[sorted_cols]:
      # We aren't unique, we're isomorphic to some other graph.
      col_matcher[sorted_cols][evs].append(next_index)
      if u[0]:
        u = [False, col_matcher[sorted_cols][evs]]
      else:
        u.append(col_matcher[sorted_cols][evs])
    else:
      # We might be unique.
      row_matcher[sorted_cols][evs] = [next_index]
  else:
    # We might be unique.
    col_matcher[sorted_cols] = {evs : [next_index]}

  unique.append(u)

unlabeled = filter(lambda x: unique[x][0], range(len(matrices)))

print "The number of DAGs with " + str(n) + " unlabeled vertices is " + \
      str(len(unlabeled))
print

matrix_writer = csv_writer(stdout, lineterminator = '\n')

for i in range(len(matrices)):
  #if unique[i][0]:
    print "Matrix", i

    for row in matrices[i]:
      matrix_writer.writerow(row)

    print unique[i][0], unique[i][1], eigenvalues[i]
    print

