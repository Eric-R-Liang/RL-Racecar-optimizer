import numpy as np

import car
from game import Game


class Car(Game):
    fineness = 3

    def __init__(self, trackcsv, carcsv):
        self.track = np.loadtxt(open(trackcsv, "rb"), delimiter=",", skiprows=1)
        self.carsim = car.car(carcsv)
        self.inplay = True
        self.distance = 0

        f = sorted(list(set(list(range(0, self.carsim.breaking, -1 * self.fineness)) + \
                               list(range(0, self.carsim.acceleration, 1 * self.fineness)))))

        self.actual_actions = sorted([(x, 0) for x in f] + [(x, 1) for x in f])
        self.possible_actions = [i for i in range(len(self.actual_actions))]
        self.len_possible_actions = len(self.possible_actions)

        self.history = []

        print(self.actual_actions)

    @property
    def nb_actions(self):
        return self.len_possible_actions

    def get_distance(self):
        return self.distance

    def get_history(self):
        return self.history

    def reset(self):
        self.carsim.car_reset()
        self.inplay = True
        self.distance = 0
        self.history = []

    def play(self, action):
        self.history.append(self.actual_actions[self.possible_actions[action]])
        self.inplay, self.distance = self.carsim.evolve(self.track[self.distance],
                                         self.actual_actions[self.possible_actions[action]],
                                         self.distance)

    def get_score(self):
        if self.carsim.time == 0:
            return 0

        score = self.distance / self.carsim.time + self.distance ** 2
        normalized = score if self.inplay else -1 / (score + 1)
        return normalized + 1

    def is_over(self):
        return (not self.inplay) or self.is_won()

    def is_won(self):
        return self.distance + 1 >= len(self.track)

    def get_state(self):
        ret = self.carsim.get_state_vector() + [
            self.distance
        ]
        ret.extend(self.track[self.distance : self.distance + 20])

        return np.array(ret)

    def get_possible_actions(self):
        return self.possible_actions


def simulate(carfile, track, inst):
    instructions = np.loadtxt(open(inst, "rb"), delimiter=",", skiprows=1)
    track = np.loadtxt(open(track, "rb"), delimiter=",", skiprows=1)
    carsim = car(carfile)

    distance = 0
    while carsim.evolve(track[distance], instructions[distance]):
        print(distance, carsim.cur_v)
        distance += 1

    print("GOT TO: " + str(distance))

# simulate("data/sample_car.csv", "data/track_1.csv", "data/sample_instructions.csv")
