import trottersuzuki as ts
import pandas as pd
import numpy as np
import pickle
import math

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt


class config:
  # -------- Trotter Suzuki ----------------- #
  dim = 512  # dimensions of grid
  radius = 24  # radius of particle?
  angular_momentum = 1
  time_step = 1e-4
  iterations = 10000  # number of iterations of evolution
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]
  coupling_vars = np.random.uniform(0, 100, (500,))  # variable `g` values
  coupling = 120.
  
  
  def wave_function(x, y):  # a working wave function
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):  # a working potential
    return 0.5 * (x ** 2 + y ** 2)


def sim2(coupling, config, omega=0):
  # Set up lattice
  grid = ts.Lattice1D(config.dim, config.radius)
  # initialize state
  state_1 = ts.GaussianState(grid, 1.)  # Create first-component system's state
  state_2 = ts.GaussianState(grid, 1.)  # Create second-component system's state
  # init potential
  potential_1 = ts.Potential(grid)
  potential_1.init_potential(config.potential_fn)  # harmonic potential
  potential_2 = ts.Potential(grid)
  potential_2.init_potential(config.potential_fn)  # harmonic potential
  # build hamiltonian with coupling strength `g1`, `g2`, `g12`
  hamiltonian = ts.Hamiltonian2Component(grid, potential_1, potential_2,
    _coupling_a=coupling[0], coupling_ab=coupling[1],
    _coupling_b=coupling[2], _omega_r=omega)
  # setup solver
  solver = ts.Solver(grid, state_1, hamiltonian, config.time_step, State2=state_2)
  # get iterations
  iterations = config.iterations
  # Evolve the system
  solver.evolve(iterations, True)
  # Compare the calculated wave functions w.r.t. groundstate function
  psi1 = state_1.get_particle_density()
  psi2 = state_2.get_particle_density()
  assert psi1.shape == (1, config.dim)
  assert psi2.shape == (1, config.dim)
  # psi / psi_max
  psi1 = psi1[0] / max(psi1[0])
  psi2 = psi2[0] / max(psi2[0])
  # save data
  return pd.DataFrame({
    'x' : grid.get_x_axis(),
    'g11' : np.ones(psi1.shape) * coupling[0],
    'g12' : np.ones(psi1.shape) * coupling[1],
    'g22' : np.ones(psi1.shape) * coupling[2],
    'psi1' : psi1,
    'psi2' : psi2,
    'omega' : np.ones(psi1.shape) * omega
    })


def sim(coupling, config):
  # get coupling strength
  # coupling = config.coupling
  # Set up lattice
  grid = ts.Lattice1D(config.dim, config.radius)
  # initialize state
  state = ts.State(grid, config.angular_momentum)
  state.init_state(config.wave_function)
  # init potential
  potential = ts.Potential(grid)
  potential.init_potential(config.potential_fn)  # harmonic potential
  # build hamiltonian with coupling strength `g`
  hamiltonian = ts.Hamiltonian(grid, potential, 1., coupling)
  # setup solver
  solver = ts.Solver(grid, state, hamiltonian, config.time_step)

  iterations = config.iterations
  # Evolve the system
  solver.evolve(iterations, False)
  # Compare the calculated wave functions w.r.t. groundstate function
  # psi = np.sqrt(state.get_particle_density()[0])
  psi = state.get_particle_density()[0]
  # psi / psi_max
  psi = psi / max(psi)
  # save data
  df = pd.DataFrame({
  'x' : grid.get_x_axis(),
  'g' : np.ones(psi.shape) * coupling,
  'psi' : psi
  })
  return df


def harmonic(x, y):
  return 0.5 * (x ** 2 + y ** 2)


def double_well_potential(x, y):
  return (4. - x**2) ** 2 / 2.


def optical_lattice_potential(x, y):
  return (x ** 2) / ( 2 + 12 * (math.sin(4 * x) ** 2) )


# def get_optical_lattice_config2():
#   config2.potential_fn = optical_lattice_potential
#   return config


# def get_double_well_config2():
#   config2.potential_fn = double_well_potential
#   return config


# def get_harmonic_config2():
#   config2.potential_fn = harmonic
#   return config2


def get_optical_lattice_config():
  config.potential_fn = optical_lattice_potential
  return config


def get_double_well_config():
  config.potential_fn = double_well_potential
  return config


def get_harmonic_config():
  config.potential_fn = harmonic
  return config


class GPApproximation:
  """Gaussian Process Wrapper

  config : Configuration; Optional [None]
  """

  def __init__(self, components=1):
    kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
    if components == 2:
      kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5, 5, 5], (1e-2, 1e2))
    self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    self.config = config

  def fit(self, X, y):
    self.gp.fit(X, y)

  def evaluate(self, X, y):
    self.error = ((self.gp.predict(X) - y) ** 2).sum() / len(y)
    return self.error

  def predict(self, X):
    y_pred, sigma = self.gp.predict(X, return_std=True)
    sigma[sigma < 0.] = 0.
    return y_pred, sigma

  def save(self, name):
    pickle.dump(self, open(name, 'wb'))

  def load(self, name):
    gps = pickle.load(open(name, 'rb'))


def make_dataset(dataframes, tr_n=20000, te_n=3000, va_n=3000):
  data_set_size = tr_n + te_n + va_n

  samples_per_df = data_set_size // len(dataframes) 
  dataset = pd.DataFrame()
  for df in dataframes:
      dataset = dataset.append(df.sample(samples_per_df), ignore_index=True)
  # shuffle
  dataset = dataset.sample(frac=1).reset_index(drop=True)
  return (
      dataset[:tr_n],
      dataset[tr_n: tr_n + te_n].reset_index(drop=True),
      dataset[tr_n + te_n : ].reset_index(drop=True)
  )


def get_closest_sim(dataframes, g):
    return min(dataframes, key=lambda df : abs(df.g[0] - g))


def get_closest_points(dataset, g=None, x=None, n=10):
  if g is not None:
    dataset = dataset.iloc[(dataset.g - g).abs().argsort()[:1000]]
  if x is not None:
    dataset = dataset.iloc[(dataset.x - x).abs().argsort()[:n]]
  return dataset.sample(n)


def get_within_range(dataset, g_low=None, g_high=None, x_low=None, x_high=None, n=10):
  if g_low is not None and g_high is not None:
    dataset = dataset[(dataset.g >= g_low) & (dataset.g <= g_high)]
  if x_low is not None and x_high is not None:
    dataset = dataset[(dataset.x >= x_low) & (dataset.x <= x_high)]
  return dataset.sample(n)



def plot_gp(x, y, y_pred, sigma):
  fig = plt.figure(figsize=(8, 6))
  plt.xticks(np.arange(-12, max(x)+1, 6.0))
  plt.xlim([-12.01, 12.01])
  plt.ylim([-0.4, 1.2])
  plt.xlabel('x', fontsize=23)
  plt.plot(x, y, c='#707CD5', alpha=0.6)
  plt.scatter(x, y_pred, s=4, c='#854EC0', alpha=0.7)
  plt.fill(np.concatenate([x, x[::-1]]),
           np.concatenate([y_pred - 1.9600 * sigma,
                          (y_pred + 1.9600 * sigma)[::-1]]),
           alpha=.3, fc='#ED358E', ec='None', label='95% confidence interval')
  plt.ylabel('$\psi$', fontsize=20)
  plt.title('Wave Function', fontsize=20)
  return fig


def cam_plot_gp(fig, x, y, y_pred, sigma, candid):
  plt.xticks(np.arange(-12, max(x)+1, 6.0))
  plt.xlim([-12.01, 12.01])
  plt.ylim([-0.4, 1.2])
  plt.xlabel('x', fontsize=23)
  # idx = np.where(x == np.array(candid))
  idx = np.in1d(x, np.array(candid)).nonzero()[0]
  idx_last = np.in1d(x, np.array([candid[-1]])).nonzero()[0]
  plt.scatter(x[idx], y[idx], s=60, c='red', alpha=0.7)
  plt.scatter(x[idx_last], y[idx_last], s=230, facecolors='none', edgecolors='r',
              linestyle='dashdot')
  # plt.axvline(x=x[idx])
  plt.plot(x, y, c='#707CD5', alpha=0.6)
  plt.scatter(x, y_pred, s=4, c='#854EC0', alpha=0.7)
  plt.fill(np.concatenate([x, x[::-1]]),
           np.concatenate([y_pred - 1.9600 * sigma,
                          (y_pred + 1.9600 * sigma)[::-1]]),
           alpha=.3, fc='#ED358E', ec='None', label='95% confidence interval')
  plt.ylabel('$\psi$', fontsize=20)
  plt.title('Wave Function', fontsize=20)


def cam_plot_gx_gp(fig, x, y, y_pred, sigma):
  pass


def partition(arr, k, min_elems_in_partition=2):
  n = arr.shape[0]
  if n % k == 0:
    return np.split(arr, k)
  if n % k >= n // k:
    k = n // min_elems_in_partition
    return np.split(arr, k)
  len_arr_1 = (n // k) * k
  p = list(np.split(arr[:len_arr_1], k))
  p.append(arr[len_arr_1:])
  return p


def choose_candidate(sigma, n_partitions, candidates_g, candidates_x):
  # partition sigma
  _p_sigma = partition(sigma, n_partitions)
  # score partitions
  scores = [ sum(sp) for sp in _p_sigma ]
  # get highest scoring idx
  winner = np.argmax(scores)
  # sample f
  # partition candidates
  _p_candidates_g = partition(candidates_g, n_partitions)
  # choose from candidates
  candidate_zone = _p_candidates[winner]
  # random sample a candidate
  return np.random.choice(candidate_zone)
