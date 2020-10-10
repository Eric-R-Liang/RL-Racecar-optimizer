import numpy as np

MAP_GAS = [-1, 500, 750, 1000, 1250, 1500]
MAP_TIRE = [-1, 500, 750, 1000, 1250, 1500]
MAP_HANDLING = [-1, 9, 12, 15, 18, 21]
MAP_SPEED = [-1, 10, 20, 30, 40, 50]
MAP_ACCEL = [-1, 10, 15, 20, 25, 30]
MAP_DECEL = [-1, -10, -15, -20, -25, -30]


class car:
    def __init__(self, csv):
        parsed = np.loadtxt(open(csv, "rb"), dtype=int, delimiter=",", skiprows=1)
        self.tire = MAP_TIRE[parsed[0]]
        self.gas = MAP_GAS[parsed[1]]
        self.handling = MAP_HANDLING[parsed[2]]
        self.speed = MAP_SPEED[parsed[3]]
        self.acceleration = MAP_ACCEL[parsed[4]]
        self.breaking = MAP_DECEL[parsed[5]]

        self.cur_v = 0
        self.cur_gas = self.gas
        self.cur_tire = self.tire
        self.time = 0

    def get_state_vector(self):
        return [
            self.cur_v,
            self.cur_gas,
            self.cur_tire,
            self.time
        ]

    def car_reset(self):
        self.cur_v = 0
        self.cur_gas = self.gas
        self.cur_tire = self.tire
        self.time = 0

    def print_error(self, *args):
        pass

    # inst = [acc, pit_stop]
    def evolve(self, radius, inst, cur_position):
        v0 = self.cur_v

        if inst[0] > self.acceleration:
            self.print_error("ERROR: illegal acceleration value: ", inst[0], cur_position)
            return False, cur_position

        if inst[0] < self.breaking:
            self.print_error("ERROR: illegal breaking value: ", inst[0], cur_position)
            return False, cur_position

        if v0 <= 0 and inst[0] <= 0:
            self.print_error("Error: stalling at one point without moving nor acceleration. inst:", inst, cur_position)
            return False, cur_position

        # resultant speed positive
        if v0 * v0 + 2 * inst[0] < 0:
            v1 = 0
        else:
            v1 = np.sqrt(v0 * v0 + 2 * inst[0])

        # no gas and accelerating
        if self.cur_gas <= 0 and inst[0] > 0:
            self.print_error("Error: no gas and accelerating", self.cur_gas, inst[0], cur_position)
            return False, cur_position

        # find vmax
        if radius != -1:
            v_max = np.sqrt(radius * self.handling / 1000000)
            if v0 > v_max or v1 > v_max:  # exceeds max
                self.print_error("Error: Speed exceeds max", v0, v1, cur_position)
                return False, cur_position

        # pitstop
        if inst[1] == 1:
            # condition 1
            if v0 != 0:
                if v0 * v0 / 2 > np.abs(self.breaking):
                    self.print_error("Error: pitstop and unable to break (too fast 1)", cur_position)
                    return False, cur_position
                t1 = 2 / v0
            else:
                t1 = 0

            # condition 2
            if v1 != 0:
                if v1 * v1 / 2 > self.acceleration:
                    self.print_error("Error: pitstop and unable to read target velocity (too fast 2)", cur_position)
                    return False, cur_position
                t2 = 2 / v1
            else:
                t2 = 0

            # 30 (pitstop time) + t1 + t2
            self.time += 30 + t1 + t2

            # restore car conditions
            self.cur_gas = self.gas
            self.cur_tire = self.tire
        else:
            if inst[0] == 0:
                self.time += 1 / v0
            else:
                self.time += (v1 - v0) / inst[0]

        self.cur_v = v1

        if self.cur_v < 0:
            self.print_error("ERROR: negative current velocity:", self.cur_v, cur_position)

        if inst[0] > 0:
            self.cur_gas -= 0.1 * inst[0] ** 2
        elif inst[0] < 0:
            self.cur_gas -= 0.1 * np.abs(inst[0]) ** 2

        if self.cur_tire < 0:
            self.print_error("Error: tire died", cur_position)
            return False, cur_position

        if inst[1] == 1:
            return True, cur_position + 2
        else:
            return True, cur_position + 1
