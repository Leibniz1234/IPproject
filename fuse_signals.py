import numpy as np

import kalman_filters


class FuseSignal:
    def __init__(self, signals, timestamps):
        self.signals = signals
        self.timestamps = timestamps

    def kalman_filter(self, filter, ini_pos):
        # filter = kf.ConstantVelocityKalmanFilter()

        fused_signal = [ini_pos]
        filter.initialize_filter(ini_pos)
        time_step = filter.dt
        print(len(self.signals))
        video_time = int((self.timestamps[0] * len(self.signals[0])) / time_step)
        for i in range(1, video_time):
            fused_signal.append(filter.predict())
            for j in range(len(self.signals)):
                multiple = (i - 1) * time_step / self.timestamps[j]
                if multiple - int(multiple) == 0 and self.signals[j]:
                    filter.update(self.signals[j][int(multiple)])
        return fused_signal

    def particle_filter(self, filter, ini_pos):
        # filter = kf.ConstantVelocityKalmanFilter()
        kalman_filtered = []
        kalman = kalman_filters.ConstantVelocityKalmanFilter(state_dim=4, obs_dim=4, dt=1/15)
        fused_signal = [ini_pos[0:2]]
        filter.initialize_filter(ini_pos[0:2])
        kalman.initialize_filter(ini_pos, init_noise_var=0.1, process_noise_var=0.01, obs_noise_var=0.1)
        time_step = filter.dt
        video_time = int((self.timestamps[0] * self.signals.shape[0]) / time_step)
        for i in range(1, video_time):
            # fused_signal.append(kalman.predict()[0:2])
            fused_signal.append(filter.predict())
            # TODO: at each timestep -> update kalman for concatenated signals (xyz,xyz) -> predict kalman w/ dt as timestamp -> run particle filter w/ velocity from predicted kalman
            kalman.update(self.signals[i])

            pred = kalman.predict()
            kalman_filtered.append(pred)
            # filter.update(self.signals[i][0:2], pred)
            filter.update(pred[0:2], pred)


        kalman.plot(plot_type="kalman_state", win_len=100)
        return fused_signal, kalman_filtered
