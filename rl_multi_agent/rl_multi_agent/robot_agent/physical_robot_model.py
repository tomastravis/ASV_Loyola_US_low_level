import numpy as np

# Surge or forward motion
# Sway or lateral displacement
# Yaw or rotation angle of the bow

class PhysicalRobotModel:
    """
    Robot physical model according to state
    X = [x_est, y_est, psi_est, u_est, v_est, r_est]
    """
    mass = 23.8 # Robot mass in kg

    rotationalInertia = 1.76 # Moment of inertia of the robot in kg*m^2 (yaw)

    xG = 0.046 # Center of gravity slightly offset

    X_udot =  -2 # Resistance of water to linear acceleration in the surge direction
    Y_vdot = -10 # Same for sway
    Y_rdot = 0 # Added mass coupling between yaw and sway. Indicates how yaw motion affects the sway hydrodynamic forces.
    N_vdot = 0 # Represents how sway motion affects the yaw hydrodynamic forces.
    N_rdot = -1 # Resistance of water to rotational acceleration in yaw.

    x_u = -0.72253 # Linear drag coefficient in surge
    x_u_abs_u = -1.32742 # Nonlinear drag forces due to water resistance, proportional to the square of velocity.
    y_v = -0.88965
    y_v_abs_v = -36.47287
    y_v_abs_r = -0.805
    y_r = -7.25
    y_r_abs_v = -0.845
    y_r_abs_r = -3.45
    n_v = 0.0313
    n_v_abs_v = 3.95645
    n_v_abs_r = 0.13
    n_r = -1.9
    n_r_abs_v = 0.08
    n_r_abs_r = -0.75
    c = np.zeros((3,3), dtype=np.float32)

    # Importance weights for error
    k_v=2.75  # If increased, direction to desired point becomes more important
    k_d=2  # If increased, distance to desired point becomes more important


    # TODO: In ROS we will need an auxiliary function for position initialization because we won't be able to choose it
    def __init__(self, id, x_ini=None, dt = 0.05, lm=5):
        self.x = x_ini if x_ini is not None else np.array([[2], [2], [np.pi/4], [0], [0], [0]], dtype=np.float32)
        self.dt = dt
        self.id = id # Robot identifier (cannot be defined, it is what it is)
        self.lm = lm # In the paper it's l, formation length with respect to the formation centroid
        self.beta_m=None # In the paper it's beta, formation angle with respect to the x-axis (If 0, it should be right in front of the centroid) North to east equals positive rotation

        # Mass or effective inertia matrix
        # Represents the robot's resistance to linear and rotational accelerations
        # in surge (x-direction), sway (y-direction), and yaw (rotation about z-axis).
        self.m = np.array([[self.mass - self.X_udot, 0, 0],
                           [0, self.mass - self.Y_vdot, self.mass*self.xG - self.Y_rdot],
                           [0, self.mass*self.xG - self.N_vdot, self.rotationalInertia - self.N_rdot]])
        self.invM = np.linalg.inv(self.m)
        # self.x_u =  0
        # self.x_u_abs_u =  0
        # self.y_v =  0
        # self.y_v_abs_v =  0
        # self.y_v_abs_r =  0
        # self.y_r =  0
        # self.y_r_abs_v =  0
        # self.y_r_abs_r =  0
        # self.n_v =  0
        # self.n_v_abs_v =  0
        # self.n_v_abs_r =  0
        # self.n_r = 0
        # self.n_r_abs_v =  0
        # self.n_r_abs_r =  0


    # Returns the robot observation (position and velocity)
    # Noise can be added to the observation and noise variance can be specified
    def observe(self, add_noise=False, sigma2=0.01):
        return self.x.T[0]

    @staticmethod
    # Translates agent action to force and rotation moment for the robot
    def get_force_tau(action):
        '''                            action intervals
        action 0 is the surge force [0, 2] (1 as minimum, why?)
        action 1 is the rotation moment [-0.5, 0.5]
        '''
        return np.vstack([action[0] + 1, 0, action[1]/2.3])

    # Simulate robot movement.
    # Calculates how the robot state (position, orientation and velocities)
    # changes under the action of forces and moments provided by action.
    # This is a simulator of robot dynamics. For ROS tasks we will only make evolve send action data.
    def evolve(self, action): # specific for that robot

        for _ in range(4):
            _, _, psi, u_r, v_r, r = self.x.flat
            ROT = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])

            c13 = -self.m[1, 1] * v_r - self.m[1, 2]*r
            c23 = self.m[0, 0] * u_r
            self.c[0, 2] = c13
            self.c[1, 2] = c23
            self.c[2, 0] = -c13
            self.c[2, 1] = -c23

            abs_nu = np.abs(self.x[3:].flat)
            D = np.array([
                [-self.x_u - self.x_u_abs_u *abs_nu[0], 0, 0 ],
                [0, -self.y_v - self.y_v_abs_v *abs_nu[1] - self.y_v_abs_r*abs_nu[2], - self.y_r - self.y_r_abs_v*abs_nu[1] - self.y_r_abs_r*abs_nu[2]],
                [0, -self.n_v - self.n_v_abs_v *abs_nu[1] - self.n_v_abs_r*abs_nu[2], - self.n_r - self.n_r_abs_v*abs_nu[1] - self.n_r_abs_r*abs_nu[2]]
                ])
            eta_dot = np.matmul(ROT, self.x[3:])# variation of position
            nu_dot = np.matmul(self.invM, self.get_force_tau(action) - np.matmul(self.c+D, self.x[3:]))# speed variation

            # Update the robot state with new position and velocity after the action has taken place
            xdot = np.vstack([eta_dot, nu_dot])
            self.x += xdot*self.dt

            # -180 180 (Angle normalization)
            self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

    # Distance between the robot and a given point
    def distanceTo(self, point):
        return np.linalg.norm(np.subtract(self.x[:2], point))

    # Calculates the expected position of a vehicle in formation
    # (or reference) based on a leader point and formation line angle
    def expected_position(self, x_v,  slope):
        '''
        x_v: [x, y]
        slope: angle of the line
        '''
        # changed l-->lm, beta-->beta_m
        return x_v + self.lm*np.array([np.cos(slope + self.beta_m), np.sin(slope + self.beta_m)])
    # formation error

    # Calculates the formation error between the robot and a reference point (expected position)
    def error_f(self, x_v, slope):
        '''
        x_v: [x, y]
        slope: angle of the line
        x_t: [x, y]
        '''
        return np.linalg.norm(self.expected_position(x_v, slope) - self.x[:2].flat)

# verify existence
    # The following function is replaced
    # def Rv(self):
    #     return self.k_v*(self.x[4]*np.cos(self.beta) - self.x[4]*np.sin(self.beta))
    # Now the function also depends on x_v and slope, so these data arguments will need to be added
    # Reward for velocity (not position)
    def Rv(self, x_v, slope):
        x_p1 = self.expected_position(x_v, slope) - self.x[:2]
        angle = np.arctan2(x_p1[1], x_p1[0]) - self.x[2]
        return self.k_v*(self.x[3]*np.cos(angle) - (np.abs(self.x[4]) + np.abs(self.x[5]))*np.abs(np.sin(angle)))  #
                              # x[3] (surge) large and cos small (better). x[4] (sway) and x[5] (yaw) small and sin large (better)

    # Reward for distance (position)
    def Rd(self,x_v, slope):
        err_max=10
        return self.k_d*-self.error_f(x_v,slope)/err_max
        # negative to subtract reward if it's too large

# Reward and penalty

