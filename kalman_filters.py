import numpy as np
import matplotlib.pyplot as plt


# Notes:
# process_noise will have to be converted to a vector for each process

class KalmanFilter:
    def __init__(self, dt=0.01, state_dim=6, obs_dim=3, control_dim=3, save_logs=True):
        """
        Kalman Filter with base class
        https://balzer82.github.io/Kalman/
        """
        self.X = np.zeros((state_dim, 1))
        self.dt = dt
        self.timestep = 0.0
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.save_logs = save_logs

        # Initial state, process and observation noise variances
        self.init_noise_var = 0.0
        self.process_noise_var = 0.0
        self.obs_noise_var = 0.0

        # State-transition matrix (6x6): x(t) = A * x(t-1) + w(t-1)
        self.A = np.eye(self.state_dim)

        # Control matrix (6x3)
        self.B = np.zeros((self.state_dim, self.control_dim))

        # Initial uncertainty (6x6)
        self.P = np.eye(self.state_dim)

        # Covariance of the process noise (6x6)
        self.Q = np.eye(self.state_dim)

        # Observation matrix (3x6) - positions are observed and not velocities
        self.H = np.zeros((self.obs_dim, self.state_dim))

        # Covariance of the observation noise (3x3)
        self.R = np.eye(self.obs_dim)

        # Kalman Gain
        self.K = np.zeros((state_dim, obs_dim))

        # Identity matrix
        self.I = np.eye(self.state_dim)

        # Create data logger
        if self.save_logs:
            self.logger = KalmanFilterLogger()

    def initialize_filter(self, init_state, init_noise_var=10.0, process_noise_var=0.0, obs_noise_var=10.0 ):
        # Initialize State
        self.X = init_state

        # Inizialize Control
        self.U = np.zeros(self.control_dim)

        # Initialize noise variances. TODO: generalize.
        self.init_noise_var = init_noise_var
        self.process_noise_var = process_noise_var
        self.obs_noise_var = obs_noise_var

        # Initialize noise covariance matrices. TODO: Generalize.
        self.P = self.P * self.init_noise_var
        self.Q = self.Q * self.process_noise_var
        self.R = self.R * self.obs_noise_var

    def predict(self, U=None):
        self.timestep += self.dt

        # Predict state
        if U is None:
            U = np.zeros(self.control_dim)
        self.X = self.A @ self.X + self.B @ U

        # Predict error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Update logs
        if self.save_logs:
            self.logger.log(self.timestep, self.X, self.P, self.K)

        return self.X

    def update(self, Z):
        # Update Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        # Update state estimate using new observation Z
        self.X = self.X + self.K @ (Z - self.H @ self.X)

        # Update uncertainty covariance
        self.P = (self.I - self.K @ self.H) @ self.P
        return self.X

    def plot(self, win_len=100, plot_type="kalman_state"):
        if self.save_logs:
            self.logger.plot(win_len=win_len, plot_type=plot_type)


class KalmanFilterLogger:
    def __init__(self):
        self.timesteps = []
        self.state_logger = []
        self.state_uncertainty_logger = []
        self.kalman_gain_logger = []

        # Figure
        plt.show()
        self.state_uncertainty_plot_iter = 0
        self.im = None
        self.cb = None

    def log(self, timestep, state=None, state_uncertainty=None, kalman_gain=None):
        self.timesteps.append(timestep)
        self.state_logger.append(state)
        self.state_uncertainty_logger.append(state_uncertainty)
        self.kalman_gain_logger.append(kalman_gain)

    def plot(self, win_len=100, plot_type="kalman_gain"):
        timesteps = self.timesteps[-win_len:]

        if plot_type == "kalman_gain":
            kalman_gain_logger = self.kalman_gain_logger[-win_len:]
            kx, ky, kz = [], [], []
            for K in kalman_gain_logger:
                kx.append(K[0, 0])
                ky.append(K[1, 0])
                # kz.append(K[2, 0])

            plt.figure("Kalman Gain")
            plt.cla()
            plt.plot(timesteps, kx, label="Kalman gain x")
            plt.plot(timesteps, ky, label="Kalman gain y")
            plt.plot(timesteps, kz, label="Kalman gain z")

            plt.title("Kalman Gain")
            plt.xlabel("Time steps")
            plt.ylabel("Kalman gain")
            plt.legend(loc="upper right")
        elif plot_type == "kalman_state":
            state_logger = self.state_logger[-win_len:]
            sx, sy, sdx, sdy = [], [], [], []
            for s in state_logger:
                sx.append(s[0])
                sy.append(s[1])
                sdx.append(s[2])
                sdy.append(s[3])

            plt.figure("Kalman States Pos")
            plt.cla()
            plt.plot(timesteps, sx, label="Kalman state x")
            plt.plot(timesteps, sy, label="Kalman state y")
            # plt.plot(timesteps, sz, label="Kalman state z")

            plt.title("Kalman States")
            plt.xlabel("Time steps")
            plt.ylabel("Kalman States")
            plt.legend(loc="upper right")

            plt.figure("Kalman States Vel")
            plt.cla()
            plt.plot(timesteps, sdx, label="Kalman state dx")
            plt.plot(timesteps, sdy, label="Kalman state dy")
            # plt.plot(timesteps, sz, label="Kalman state z")

            plt.title("Kalman States")
            plt.xlabel("Time steps")
            plt.ylabel("Kalman States")
            plt.legend(loc="upper right")
        elif plot_type == "kalman_state_uncertainty":
            self.state_uncertainty_plot_iter += 1
            plt.figure("Kalman State Uncertainty Covariance Matrix")
            plt.title("Kalman State Uncertainty Covariance Matrix $P$")
            if self.state_uncertainty_plot_iter == 1:
                self.im = plt.imshow(self.state_uncertainty_logger[-1])
            else:
                self.cb.remove()
                self.im.set_data(self.state_uncertainty_logger[-1])
            self.cb = plt.colorbar()

        else:
            print("[ERROR] Unrecognized plot_type")


class ConstantVelocityKalmanFilter(KalmanFilter):
    def __init__(self, dt=0.01, state_dim=4, obs_dim=2, save_logs=True):
        super().__init__(dt=dt, state_dim=state_dim, obs_dim=obs_dim, save_logs=save_logs)

        # State-transition matrix (6x6): x(t) = A * x(t-1) + w(t-1)
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Observation matrix (6x6) - positions are observed and not velocities
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]
                           ])


class ConstantVelocityKalmanFilterWithControl(KalmanFilter):
    def __init__(self, dt=0.01, state_dim=4, obs_dim=2, control_dim=2, save_logs=True):
        super().__init__(dt, state_dim, obs_dim, control_dim, save_logs)

        # State-transition matrix (6x6): x(t) = A * x(t-1) + w(t-1)
        self.A = np.array([[1, 0, 0, self.dt, 0, 0],
                           [0, 1, 0, 0, self.dt, 0],
                           [0, 0, 1, 0, 0, self.dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # Control matrix (6x3)
        self.B = np.array([[1 / 2.0 * self.dt ** 2, 0, 0],
                           [0, 1 / 2.0 * self.dt ** 2, 0],
                           [0, 0, 1 / 2.0 * self.dt ** 2],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        # Observation matrix (3x6) - positions are observed and not velocities
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


class ConstantAccelerationKalmanFilter(KalmanFilter):
    def __init__(self, dt=0.001, state_dim=9, obs_dim=3):
        super().__init__(dt, state_dim, obs_dim)

        # State vector (9x1): [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

        # State-transition matrix (9x9): x(t) = A * x(t-1) + w(t-1)
        self.A = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1 / 2.0 * self.dt ** 2, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1 / 2.0 * self.dt ** 2, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1 / 2.0 * self.dt ** 2],
                           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        # Observation matrix (3x9) - positions are observed and not velocities or accelerations
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


class ConstantAccelerationKalmanFilterExtendedObservation(KalmanFilter):
    def __init__(self, dt=0.001, state_dim=9, obs_dim=6):
        super().__init__(dt, state_dim, obs_dim)

        # State vector (9x1): [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

        # State-transition matrix (9x9): x(t) = A * x(t-1) + w(t-1)
        self.A = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1 / 2.0 * self.dt ** 2, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1 / 2.0 * self.dt ** 2, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 1 / 2.0 * self.dt ** 2],
                           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        # Observation matrix (6x9) - positions and accelerations are observed and not velocities
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
