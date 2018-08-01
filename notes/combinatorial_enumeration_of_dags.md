# Combinatorial Enumeration of Directed Acyclic Graphs

- [Sequence A003024: Number of Acyclic Digraphs with n Labeled Nodes](https://oeis.org/A003024)
- [Sequence A003087: Number of Acyclic Digraphs with n Unlabeled Nodes](https://oeis.org/A003087)
- [Weinstein's Conjecture](http://mathworld.wolfram.com/WeissteinsConjecture.html)
- [Uniform Random Generation of Large Acyclic Digraphs (arXiv, Kuipers 2013)](https://arxiv.org/pdf/1202.6590.pdf)
- [Uniform Random Generation of Large Acyclic Digraphs (Statistics and Computing, Kuipers 2013)](https://link.springer.com/article/10.1007/s11222-013-9428-y)
- [Acyclic Digraphs and Eigenvalues of (0,1)-Matrices (McKay 2004)](https://cs.uwaterloo.ca/journals/JIS/VOL7/Sloane/sloane15.html)
- [Counting Unlabeled Acyclic Digraphs (Robinson 1976)](http://doi.org/10.1007/BFb0069178)
- [Counting Labeled Acyclic Digraphs (Robinson 1973](http://www.amazon.com/exec/obidos/ASIN/012324255X/ref=nosim/ericstreasuretro)
- [Enumeration of Acyclic Diagraphs (Robinson 1970)](https://oeis.org/A003024/a003024.pdf)
- [Acyclic Orientations of Graphs (Discrete Math, Stanley 1973)](https://doi.org/10.1016/0012-365X(73)90108-8)
- [Acyclic Orientations of Graphs (MIT Website, Stanley 1973)](http://www-math.mit.edu/~rstan/pubs/pubfiles/18.pdf) 

A matrix `A` is an adjacency matrix of a DAG if and only if `A + I` is a `(0, 1)` matrix with all eigenvalues positive, where I denotes the identity matrix.

The number of DAGs with `n` labeled vertices is counted by OEIS sequence A003024:

```
a_n = \sum_{k=1}^{n} (-1)^{k-1} \binom{n}{k} 2^{k(n-k)} a_{n-k} 
```

The first few numbers of this sequence are:

```
a_1 = 1
a_2 = 3
a_3 = 25
a_4 = 543
a_5 = 29281
a_6 = 3781503
```

The number of DAGs with `n` unlabeled vertices (which is what we care about for testing) is counted by OESI sequence A003087, which does not have a concise expression as a recurrence relation. The first few numbers of this sequence are:

```
a_1 = 1
a_2 = 2
a_3 = 6
a_4 = 31
a_5 = 302
a_6 = 5984
```

The basis(?) of the set of strictly upper (or lower) triangular `(0, 1)` matrices seems to count the number of DAGs as well.

The number of strictly upper (or lower) triangular `(0, 1)` matrices is:

```
2 ^ (((n - 1) * n)) / 2)
```

This is permutation with repetition: the number of `m`-tuples formed from the 2-element alphabet `(0, 1)`.

`m` is the number of potentially non-zero entries in a strictly upper (or lower) triangular `(0, 1)` matrix. `m` is equal to the `n-1`th triangular number, e.g. `((n - 1) * n) / 2`. This is evident visually and can probably be proven by induction.

**NOTE:** All graphs are expressed as either `digraph`s in [the DOT language](https://www.graphviz.org/doc/info/lang.html) or `(0, 1)` comma separated matrices.

## 1 Vertex Unlabeled DAGs

Number: 1

```
a;
```

## 2 Vertex Unlabeled DAGs

Number: 2

```
a;
b;
```

```
a -> b;
```

## 3 Vertex Unlabeled DAGs

Number: 6

```
a;
b;
c;
```

```
a -> b;
c;
```

```
a -> c;
b -> c;
```

```
a -> b;
b -> c;
```

```
a -> b;
a -> c;
```

```
a -> b;
a -> c;
b -> c;
```

## 4 Vertex Unlabeled DAGs

Number: 31

0,0,0,0
0,0,0,0
0,0,0,0
0,0,0,0
  
  0,1,1,1
  0,0,1,1
  0,0,0,0
  0,0,0,0

  0,1,1,1
  0,0,0,1
  0,0,0,0
  0,0,0,0

  Dependent
  0,1,1,1
  0,0,1,0
  0,0,0,0
  0,0,0,0

  0,0,1,1
  0,0,1,1
  0,0,0,0
  0,0,0,0

  Dependent
  0,1,0,1
  0,0,1,1
  0,0,0,0
  0,0,0,0

  0,1,1,0
  0,0,1,1
  0,0,0,0
  0,0,0,0

  0,1,1,1
  0,0,0,1
  0,0,0,1
  0,0,0,0

  0,1,1,1
  0,0,1,0
  0,0,0,1
  0,0,0,0

  0,0,1,1
  0,0,1,1
  0,0,0,1
  0,0,0,0

  0,1,0,1
  0,0,1,1
  0,0,0,1
  0,0,0,0

  0,1,1,0
  0,0,1,1
  0,0,0,1
  0,0,0,0

0,1,1,1
0,0,1,1
0,0,0,1
0,0,0,0
