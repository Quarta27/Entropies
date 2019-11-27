# The (h, pi)-entropy class can be expressed as H(h, pi)(s) = h(sum(pi(p_i)))
# where pi(p_i) is a function applied at the probability of the layer and h
# is a function that sum each pi value and apply a series of operation that
# depends on the entropy function used.

import numpy as np
import math as mt

s = 0.5
m = 1.1

def define_s_value(val):
   """ Allows to change the value of the constant s """
   global s
   s = val

def total_degree(matrix):
   """ Performs the total degree of the input matrix """
   total = 0
   for row in matrix:
      for elem in row:
         total = total + elem
   return total

def node_degree(i, matrix):
   """ Performs the degree of node-i from given matrix """
   degree = 0
   for elem in matrix[i]:
      degree = degree + elem
   return degree

def node_probability(i, matrix):
   """ Performs the node probability trough the degrees """
   return node_degree(i, matrix) / total_degree(matrix)

def nodes_probability(matrix):
   """ Performs a list that contains a degree with his occurences """
   max_degree = (len(matrix) * 2) - 1
   node_probs = [0] * max_degree
   act_degree = 0

   for index in range(0, len(matrix) - 1):
      act_degree = node_degree(index, matrix)
      node_probs[act_degree] = node_probs[act_degree] + 1
   for i in range(0, max_degree):
      node_probs[i] = node_probs[i] / len(matrix)
   return node_probs

def arimoto_entropy(matrix):
   """ Performs Arimoto entropy given a matrix """
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = entropy + (mt.pow(prob, (1/s)))
   entropy = (mt.pow(entropy, s) - 1) / (s - 1)
   return entropy

def havrda_charvat_entropy(matrix):
   """ Performs Havrda-Charvat entropy given a matrix """
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = entropy + ((mt.pow(prob, s) - prob) / (1 - s))
   return entropy

def renyi_entropy(matrix):
   """ Performs Renyi entropy given a matrix """
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = entropy + mt.pow(prob, s)
   entropy = -(mt.log(entropy) / (s - 1)) if (s > 0 and s != 1) else 0 
   return entropy

def shannon_entropy(matrix):
   """ Performs Shannon entropy given a matrix """
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = (entropy + prob) if prob == 0 else entropy + ((-prob * mt.log(prob)))
   return entropy

def sharma_mittal_entropy(matrix):
   """ Performs Sharma-Mittal entropy given a matrix """
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = (entropy + prob) if prob == 0 else entropy + ((prob * mt.log(prob)))
   entropy = (mt.exp((s - 1) * entropy) - 1) / (1 - s)
   return entropy

def tsallis_entropy(matrix):
   """ Performs Tsallis entropy given a matrix """
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = entropy + mt.pow(prob, s)
   entropy = (entropy - 1) / (1 - s)
   return entropy

def varma_entropy(matrix):
   # r = s e m a caso
   """ Performs Varma entropy given a matrix - Not yet developed"""
   entropy = 0
   probs = nodes_probability(matrix)
   for prob in probs:
      entropy = entropy + mt.pow(prob, (s - m + 1))
   entropy = mt.log(entropy) / (m - s)
   return entropy
