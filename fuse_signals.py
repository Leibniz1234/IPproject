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
        kalman = kalman_filters.ConstantVelocityKalmanFilter()
        fused_signal = [ini_pos[0:3]]
        filter.initialize_filter(ini_pos[0:3])
        kalman.initialize_filter(ini_pos)
        time_step = filter.dt
        video_time = int((self.timestamps[0] * len(self.signals[0])) / time_step)
        for i in range(1, video_time):
            fused_signal.append(filter.predict())
            for j in range(len(self.signals)):
                multiple = (i - 1) * time_step / self.timestamps[j]
                if multiple - int(multiple) == 0 and self.signals[j]:
                    filter.update(self.signals[j][int(multiple)], np.array(kalman.X))
                    kalman.update(self.signals[j][int(multiple)])
        return fused_signal
