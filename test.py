from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import sgd
from qlearning4k.games import Catch, Snake
from qlearning4k import Agent

nb_frames = 1
grid_size = 10
hidden_size = 100

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(5))
model.compile(sgd(lr=.01), "mse")

game = Snake(grid_size)
agent = Agent(model)
agent.train(game, nb_epoch=5000)

agent.play(game)