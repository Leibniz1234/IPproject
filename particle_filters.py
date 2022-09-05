import numpy as np
import copy


def normalize_weights(weights):
    """
    Normalize all particle weights.
    """

    # Compute sum weighted samples
    sum_weights = sum(weights)

    # Check if weights are non-zero
    if sum_weights < 1e-15:
        print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(
            sum_weights))

        # Set uniform weights
        return np.ones_like(weights) / len(weights)

    # Return normalized weights
    return weights / sum_weights


def resample(weights):
    # resample the weights if needed
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * weights).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
    return indices


class ParticleFilter:
    def __init__(self, N, std_vector=None, state_dim=2, max_weigth_threshold=0.5, dt=0.01):
        """
        :param N(int): number of particles
               state_dim(int): dimensions of the states(3 for xyz)
               std_vector(np.array): standard_deviation vector for generation of particles
        """
        self.N = N
        self.particles = []
        self.state_dim = state_dim
        self.mean_state = np.zeros((state_dim,))
        self.PE_s = np.zeros((N,))
        self.weights = np.ones(self.N) / self.N
        self.dt = dt
        self.max_weight_threshold = max_weigth_threshold
        self.sampling_scale = 0.11
        if std_vector is None:
            self.std_vector = np.ones_like(self.mean_state) * 0.05
        else:
            self.std_vector = std_vector

    def generate_particles_gaussian(self, x, std_scale=0.10):
        """
        Generate particles using a Gaussian distribution. Only standard
        deviations can be provided hence the covariances are all assumed zero.
        :param x:Values to generate particles
        :param std_vector: Standard deviations (one for each dimension)
        """

        # Initialize particles with uniform weight distribution
        x = np.array(x)
        particles = []
        for i in range(self.N):
            # Get state sample

            std_vector = self.std_vector * std_scale

            noise_i = np.random.normal(np.zeros(len(std_vector)), std_vector,
                                       size=(len(std_vector)))
            x_i = x + noise_i

            # Add particle i
            particles.append(x_i)
        self.particles = particles
        self.particles = np.array(self.particles)

    def generate_particles_uniform(self, x, std_scale=0.10):
        """
        Generate particles using a Gaussian distribution. Only standard
        deviations can be provided hence the covariances are all assumed zero.
        :param x:Values to generate particles
        :param std_vector: Standard deviations (one for each dimension)
        """

        # Initialize particles with uniform weight distribution
        x = np.array(x)
        particles = []
        for i in range(self.N):
            # Get state sample

            std_vector = self.std_vector * std_scale

            noise_i = np.random.uniform(-std_vector,std_vector,
                                       size=(len(std_vector)))
            x_i = x + noise_i

            # Add particle i
            particles.append(x_i)
        self.particles = particles
        self.particles = np.array(self.particles)

    def initialize_filter(self, init_state):
        self.generate_particles_uniform(init_state)

    def predict(self):
        self.mean_state = np.sum(np.array(self.particles).T * self.weights, axis=-1).T
        return self.mean_state

    def update(self, obs, vel):
        """

        :param obs: observation(position of hand at time k)
               vel: velocity of hand at time k
        """

        # compute likelihood according to input
        # generate particles based on observation
        old_particles = copy.deepcopy(self.particles)
        self.generate_particles_gaussian(obs, std_scale=self.sampling_scale)

        # import matplotlib.pyplot as plt
        # # fig, ax = plt.subplots(figsize=(10,10))
        # plt.scatter(self.particles[:,0], self.particles[:,1], c='b')
        # plt.scatter(self.mean_state[0], self.mean_state[1], c='r')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.show()
        # calculate likelihood and update weights according to the paper
        # print(f"vel: {vel[3:6]}")
        PE_s = (np.array(self.particles) - (np.array(old_particles) + vel[2:4] * self.dt)) ** 2
        self.PE_s += PE_s.sum(axis=1)
        sigma = np.std(self.PE_s)
        weights = np.exp((-(self.PE_s - np.min(self.PE_s)) ** 2) / (2 * sigma ** 2))
        # print(f"weights before: {np.min(weights)},  {np.max(weights)}")
        weights = normalize_weights(weights)
        # print(f"weights after: {np.min(weights)},  {np.max(weights)}")
        max_weight = np.max(weights)
        n_eff = (1.0 / np.sum(self.weights ** 2))
        print(f"n_eff: {n_eff}")

        if n_eff < 1000:
            self.sampling_scale = 0.06
        # elif n_eff > 250:
        #     self.sampling_scale = 8
        # elif n_eff > 150:
        #     self.sampling_scale = 5
        # elif n_eff > 50:
        #     self.sampling_scale = 3
        if n_eff < 65:
            self.sampling_scale = 0.05
            indices = resample(weights)
            self.particles = self.particles[indices, :]
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights = weights

        print(f"sampling_scale: {self.sampling_scale}")