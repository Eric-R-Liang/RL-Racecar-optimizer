import time

from keras.models import Sequential
from keras.layers import *
from qlearning4k.games import Snake
from keras.optimizers import *
from qlearning4k import Agent

from qlearning4k.games.sim import Car

from keras import backend as K

K.set_image_dim_ordering('th')

# car_game = Car('/data/daisy/data/short.csv', '/data/daisy/data/sample_car.csv')
car_game = Car('/data/daisy/data/longer2.csv', '/data/daisy/data/sample_car.csv')
print("Car State:", car_game.get_state())

time.sleep(1)

nb_frames = 10
nb_actions = len(car_game.get_possible_actions())
linput = len(car_game.get_state())

print("get_state()", car_game.get_state())
print("get_possible_actions()", car_game.get_possible_actions())

activation_method = 'sigmoid'

print("nb_actions", nb_actions)
model = Sequential()
model.add(Flatten(input_shape=(nb_frames, linput)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation=activation_method))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation=activation_method))
model.add(Dense(nb_actions))
model.compile(RMSprop(), 'MSE')

agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
agent.train(car_game, batch_size=128, nb_epoch=5000, gamma=0.4)
agent.play(car_game)
