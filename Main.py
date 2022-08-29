from Environment import Environment
from Agent import Agent
from PrioritizedSweeping import PrioritizedSweeping

# Get input for the environment size
env_size = int(input('Please enter environment size (nxn): '))
environment = Environment(n=env_size)

learning_rate = float(input('Please enter learning rate: '))
discount_rate = float(input('Please enter discount rate: '))
epsilon = float(input('Please enter epsilon value: '))
threshold = float(input('Please enter threshold value: '))
n_value = int(input('Please enter n value: '))

prioritized_sweeping = PrioritizedSweeping(environment=environment, agent=Agent(), learning_rate=learning_rate,
                                           discount_rate=discount_rate, epsilon=epsilon, threshold=threshold, n=n_value)

# Train
num_of_episodes = int(input('Please enter number of episodes for training process: '))
prioritized_sweeping.train(num_of_episodes=num_of_episodes)

test_choice = input('Do you want to test it? (y or n): ')
if test_choice == 'y':
    prioritized_sweeping.test()
