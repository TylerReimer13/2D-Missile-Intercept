import numpy as np
import matplotlib.pyplot as plt


class Vehicle:
    def __init__(self, role='', pos=np.zeros(2), vel=np.zeros(2), accel=np.zeros(2)):
        self.role = role
        self.pos = pos
        self.vel = vel
        self.accel = accel

        self.pos_hist = self.pos.copy()
        self.vel_hist = self.vel.copy()
        self.accel_hist = self.accel.copy()

        self.N = 3.

    @property
    def states(self):
        return np.hstack((self.pos_hist, self.vel_hist, self.accel_hist))

    @property
    def curr_pos(self):
        return self.pos.copy()

    @property
    def curr_vel(self):
        return self.vel.copy()

    def plot_states(self):
        plt.plot(self.pos_hist[:, 0], self.pos_hist[:, 1], label=self.role)
        plt.legend()
        plt.grid()
        plt.show()

    def pn_guidance(self, r, vr):
        r_3d = np.zeros(3)
        r_3d[:2] = 3
        vr_3d = np.zeros(3)
        vr_3d[:2] = vr

        rotation_vec = (np.cross(r, vr_3d)) / (r @ r)
        a_cmd = self.N * np.cross(vr_3d, rotation_vec)
        return a_cmd[:2]

    def update(self, pos=None, vel=None):
        if self.role == 'Interceptor':
            rng = pos - self.pos
            cv = vel - self.vel

            pn_cmd = self.pn_guidance(rng, cv)
            self.accel = pn_cmd
        else:
            self.accel = np.zeros(2)

        self.vel += self.accel * DT
        self.pos += self.vel * DT

        self.pos_hist = np.vstack((self.pos_hist, self.pos.copy()))
        self.vel_hist = np.vstack((self.vel_hist, self.vel.copy()))
        self.accel_hist = np.vstack((self.accel_hist, self.accel.copy()))


if __name__ == "__main__":
    DT = .01
    t = 0.
    tf = 100.

    targ_pos = np.array([-50., 500.])
    targ_vel = np.array([12., 0.])
    targ_accel = np.array([0., 0.])
    target = Vehicle('Target', targ_pos, targ_vel, targ_accel)

    int_pos = np.array([0., 0.])
    int_vel = np.array([0., 10.])
    int_accel = np.array([0., 0.])
    interceptor = Vehicle('Interceptor', int_pos, int_vel, int_accel)

    while t <= tf:
        target.update()
        interceptor.update(target.curr_pos, target.curr_vel)

        if np.linalg.norm(target.curr_pos - interceptor.curr_pos) <= .5:
            print('Intercepted at: ', round(t, 2), 'seconds')
            break

        t += DT

    plt.plot(target.states[:, 0], target.states[:, 1], label="Target")
    plt.plot(interceptor.states[:, 0], interceptor.states[:, 1], label='Interceptor')
    plt.plot(interceptor.states[-1, 0], interceptor.states[-1, 1], 'rX', label='Intercept Point')
    plt.legend()
    plt.grid()
    plt.xlabel('X (m)')
    plt.ylabel('ALT (m)')
    plt.show()
