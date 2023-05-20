class PID:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.setpoint = setpoint

        self.previous_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def saturate(self, value):
        return max(0.0, min(1.0, value))

    def update(self, measurement):
        error = self.setpoint - measurement
        p = self.Kp * error
        self.integral += error * self.dt

        # Anti-windup
        self.integral = self.saturate(self.integral)
        
        i = self.Ki * self.integral
        d = self.Kd * (error - self.previous_error) / self.dt if self.dt > 0 else 0.0

        self.previous_error = error
        output = p + i + d

        # Saturate the output
        return self.saturate(output)