import numpy as np
import math
from numpy import exp 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class MaxEntropyDistribution:
  """
  Computes the max-entropy distribution on the finite interval `bnds`, satisfying `constraints`.
  """
  def __init__(self, bnds, constraints=[], dx=.1, discrete=False):
    assert len(bnds)==2 and bnds[1]>bnds[0]
    self.discrete = discrete
    if discrete:
      self.dx = 1
    else:
      self.dx = dx
    N = int((bnds[1]-bnds[0])/self.dx)
    self.domain = np.linspace(bnds[0],bnds[1],N+1)
    if len(constraints)==0:
      self.dP_list = [1/(N+1) for x in self.domain]
    else:
      self.constraints = constraints
      grad_L = lambda lamb: tuple(self._int_f(c[0],lamb) - c[1] for c in self.constraints)
      self.lamb = fsolve(grad_L, np.zeros(len(constraints)) )
      self.dP_list = [self._p(x,self.lamb) for x in self.domain]
      
    self.F_list  = [sum(self.dP_list[:i]) for i in range(len(self.domain))]

  def _helper(self,x,lamb):
    assert len(lamb)==len(self.constraints)
    return sum(l * c[0](x) for l,c in zip(lamb,self.constraints))
  
  def _Z(self,lamb):
    return sum( exp( -self._helper(x, lamb) ) for x in self.domain )

  def _p(self, x, lamb):
    return exp( -self._helper(x, lamb) ) / self._Z(lamb)

  def _int_f(self, f, lamb):
    return sum( self._p(x, lamb) * f(x) for x in self.domain )
  
  def _to_index(self, x):
    assert x >= self.domain[0] and x <= self.domain[-1]
    return int((x - self.domain[0]) // self.dx)

  def _to_x(self, i):
    assert isinstance(i, int)
    return self.domain[0]+i*self.dx

  def dP(self, x):
    """
    Returns the 'infinitesimal' probability dP(X=x). This equals the probability density function times self.dx.
    """
    return self.dP_list[self._to_index(x)]
  
  def F(self, x):
    """
    Evaluates the cumulative distribution function F at point x. Returns the probability P(X≤x).
    """
    return self.F_list[self._to_index(x)]

  def E(self, f):
    """
    Computes the expected value of a scalar function f under the computed probability measure.
    """
    return sum(f(x)*self.dP(x) for x in self.domain)

  def plot_pdf(self):
    """
    Plots the probability density function (or a probability scatter plot, in the case of a discrete distribution).
    """
    if self.discrete:
      plt.scatter(self.domain,[self.dP_list[i]/self.dx for i in range(len(self.domain))])
    else:
      plt.plot(self.domain,[self.dP_list[i]/self.dx for i in range(len(self.domain))])
    ax = plt.gca()
    ax.set_ylim([0, None])
    plt.title("PDF")
    plt.show()

  def plot_cdf(self):
    """
    Plots the cumulative distribution function.
    """
    if self.discrete:
      plt.scatter(self.domain,[self.F_list[i] for i in range(len(self.domain))])
    else:
      plt.plot(self.domain,[self.F_list[i] for i in range(len(self.domain))])
    plt.title("CDF")
    plt.show()

  def sample(self, N=1):
    """
    Generate a list of N iid samples from the computed probability distribution.
    """
    U = np.random.uniform(0,1,N)
    indices = range(len(self.domain))
    return [ self._to_x(
        max([i for i in indices if self.F_list[i] <= U[k]])
    ) for k in range(N) ]