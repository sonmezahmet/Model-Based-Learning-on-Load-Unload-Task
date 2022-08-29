import Constants
import random
from PriorityQueue import PriorityQueue
import matplotlib.pyplot as plt


class PrioritizedSweeping:
    def __init__(self, environment, agent, learning_rate, discount_rate, epsilon, threshold, n):
        self.environment = environment
        self.agent = agent
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.threshold = threshold
        self.n = n

    def train(self, num_of_episodes):
        model = dict()
        train_dict = dict()
        for i in range(num_of_episodes):
            # Initialize agent
            self.agent.current_state = self.environment.env[0][0]
            self.agent.is_loaded = False
            self.agent.is_unloaded = False
            self.agent.total_reward = 0

            # Initialize priority queue
            p_queue = PriorityQueue()

            is_episode_finished = False
            step_count = 0
            while not is_episode_finished:
                current_state = self.agent.current_state

                # Select action with e-greedy
                action = self.__select_action(state=current_state)

                # Take action and observe reward and next state
                next_state, reward, is_episode_finished = self.__take_action(state=current_state, action=action)

                # Update model
                if current_state.state_type == Constants.upper_slope:
                    key = f'{current_state.coords[0]},{current_state.coords[1]},{Constants.upper_slope},{action},' \
                          f'{self.agent.is_loaded},{self.agent.is_unloaded}'
                elif current_state.state_type == Constants.lower_slope:
                    key = f'{current_state.coords[0]},{current_state.coords[1]},{Constants.lower_slope},{action},' \
                          f'{self.agent.is_loaded},{self.agent.is_unloaded}'
                else:
                    key = f'{current_state.coords[0]},{current_state.coords[1]},{None},{action},' \
                          f'{self.agent.is_loaded},{self.agent.is_unloaded}'
                if action == Constants.load_action:
                    model[key] = [next_state, reward, True, False]
                elif action == Constants.unload_action:
                    model[key] = [next_state, reward, True, True]
                else:
                    model[key] = [next_state, reward, self.agent.is_loaded, self.agent.is_unloaded]

                # Get appropriate Q tables
                if self.agent.is_loaded is False and self.agent.is_unloaded is False:
                    q_table = current_state.q0
                    if action == Constants.load_action:
                        q_table_next = next_state.q1
                    else:
                        q_table_next = next_state.q0
                elif self.agent.is_loaded is True and self.agent.is_unloaded is False:
                    q_table = current_state.q1
                    if action == Constants.unload_action:
                        q_table_next = next_state.q2
                    else:
                        q_table_next = next_state.q1
                else:
                    q_table = current_state.q2
                    q_table_next = next_state.q2

                # Calculate probability
                p = abs(reward + self.discount_rate * max(q_table_next.values()) - q_table[action])

                # If p > threshold, then insert s,a into PQueue with priority p
                if p > self.threshold:
                    p_queue.insert([p, current_state, action, self.agent.is_loaded, self.agent.is_unloaded])

                # Repeat N times, while not isEmpty(PQueue)
                for j in range(self.n):
                    if not p_queue.isEmpty():
                        p_state, p_action, is_loaded, is_unloaded = p_queue.pop()
                        if p_state.state_type == Constants.upper_slope:
                            key = f'{p_state.coords[0]},{p_state.coords[1]},{Constants.upper_slope},{p_action},' \
                                  f'{is_loaded},{is_unloaded}'
                        elif p_state.state_type == Constants.lower_slope:
                            key = f'{p_state.coords[0]},{p_state.coords[1]},{Constants.lower_slope},{p_action},' \
                                  f'{is_loaded},{is_unloaded}'
                        else:
                            key = f'{p_state.coords[0]},{p_state.coords[1]},{None},{p_action},' \
                                  f'{is_loaded},{is_unloaded}'
                        m_state, m_reward, m_is_loaded, m_is_unloaded = model[key]

                        # Get appropriate Q tables
                        if is_loaded is False and is_unloaded is False:
                            p_q_table = p_state.q0
                            if m_is_loaded is False and m_is_unloaded is False:
                                p_q_table_next = m_state.q0
                            elif m_is_loaded is True and m_is_unloaded is False:
                                p_q_table_next = m_state.q1
                            else:
                                p_q_table_next = m_state.q2
                        elif is_loaded is True and is_unloaded is False:
                            p_q_table = p_state.q1
                            if m_is_loaded is False and m_is_unloaded is False:
                                p_q_table_next = m_state.q0
                            elif m_is_loaded is True and m_is_unloaded is False:
                                p_q_table_next = m_state.q1
                            else:
                                p_q_table_next = m_state.q2
                        else:
                            p_q_table = p_state.q2
                            if m_is_loaded is False and m_is_unloaded is False:
                                p_q_table_next = m_state.q0
                            elif m_is_loaded is True and m_is_unloaded is False:
                                p_q_table_next = m_state.q1
                            else:
                                p_q_table_next = m_state.q2

                        # Update q table
                        td_error = m_reward + self.discount_rate * max(p_q_table_next.values()) - p_q_table[p_action]
                        p_q_table[p_action] = p_q_table[p_action] + self.learning_rate * td_error

                        # Find all s", a" which lead to s from model
                        lead_list = []
                        for model_key in model:
                            split_model_key = model_key.split(',')
                            if model[model_key][0] == p_state:
                                if model[model_key][2] is is_loaded and model[model_key][3] is is_unloaded:
                                    s_coord_x = int(split_model_key[0])
                                    s_coord_y = int(split_model_key[1])
                                    s_slope_type = split_model_key[2]
                                    s_action = split_model_key[3]
                                    s_is_loaded = True if split_model_key[4] == 'True' else False
                                    s_is_unloaded = True if split_model_key[5] == 'True' else False
                                    if s_slope_type == Constants.upper_slope:
                                        s_double_prime = self.environment.env[s_coord_x][s_coord_y].upper_slope
                                    elif s_slope_type == Constants.lower_slope:
                                        s_double_prime = self.environment.env[s_coord_x][s_coord_y].lower_slope
                                    else:
                                        s_double_prime = self.environment.env[s_coord_x][s_coord_y]
                                    temp_reward = model[model_key][1]
                                    temp_list = [s_double_prime, s_action, temp_reward, s_is_loaded, s_is_unloaded]
                                    lead_list.append(temp_list)

                        # Repeat for all s”,a” (from within all previously experienced pairs) predicted to lead to s
                        for k in lead_list:
                            predicted_reward = k[2]

                            # Get appropriate Q table
                            if k[3] is False and k[4] is False:
                                q_table_lead = k[0].q0
                            elif k[3] is True and k[4] is False:
                                q_table_lead = k[0].q1
                            else:
                                q_table_lead = k[0].q2

                            # Calculate p_lead
                            p_lead = abs(
                                predicted_reward + self.discount_rate * max(p_q_table.values()) - q_table_lead[k[1]])

                            # If p_lead > threshold, then insert s,a into PQueue with priority p
                            if p_lead > self.threshold:
                                p_queue.insert([p_lead, k[0], k[1], k[3], k[4]])

                # Move agent to next state
                if action == Constants.load_action:
                    self.agent.is_loaded = True
                elif action == Constants.unload_action:
                    self.agent.is_unloaded = True
                self.agent.previous_state = self.agent.current_state
                self.agent.current_state = next_state
                self.agent.total_reward += reward

                step_count += 1

            train_dict[i] = step_count
            print(f'Episode {i} is finished in {step_count} steps. Agent\'s total reward is {self.agent.total_reward}.')
            print('-' * 100)

        # Plot train results
        episodes = list(train_dict.keys())
        steps = list(train_dict.values())
        plt.plot(episodes, steps)
        plt.title('Prioritized Sweeping Training')
        plt.xlabel('Episode')
        plt.ylabel('Step Count')
        plt.show()

    def test(self):
        # Initialize the agent
        self.agent.current_state = self.environment.env[0][0]
        self.agent.previous_state = None
        self.agent.is_loaded = False
        self.agent.is_unloaded = False
        self.agent.total_reward = 0

        is_episode_finished = False
        step_count = 0
        while not is_episode_finished:
            # Select an action
            action = self.__select_action(state=self.agent.current_state, greedy_selection=True)

            # Take action; observe reward and next state
            next_state, reward, is_episode_finished = self.__take_action(state=self.agent.current_state, action=action)

            print(f'Step: {step_count}, Current State: {self.agent.current_state.coords}, Action: {action}, '
                  f'Next State: {next_state.coords}, Reward: {reward}')

            # Move agent to next state
            self.agent.total_reward += reward
            if action == Constants.load_action:
                self.agent.is_loaded = True
            elif action == Constants.unload_action:
                self.agent.is_unloaded = True
            self.agent.previous_state = self.agent.current_state
            self.agent.current_state = next_state

            step_count += 1

        print(f'Test is finished in {step_count} steps. Total Reward: {self.agent.total_reward}')

    def __select_action(self, state, greedy_selection=False):
        current_state = state

        if greedy_selection:
            # Take greedy action
            if self.agent.is_loaded is False and self.agent.is_unloaded is False:
                # Consider q0 table
                q_table = current_state.q0
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is False:
                # Consider q1 table
                q_table = current_state.q1
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is True:
                # Consider q2 table
                q_table = current_state.q2
                return max(q_table, key=q_table.get)

        dice = random.uniform(0, 1)
        if dice < self.epsilon:
            # Take random action
            if current_state.state_type != Constants.load_state and current_state.state_type != Constants.unload_state:
                return random.choice(current_state.get_possible_actions())
            elif current_state.state_type == Constants.load_state:
                return random.choice(current_state.get_possible_actions(is_loaded=self.agent.is_loaded,
                                                                        is_unloaded=self.agent.is_unloaded))
            elif current_state.state_type == Constants.unload_state:
                return random.choice(current_state.get_possible_actions(is_loaded=self.agent.is_loaded,
                                                                        is_unloaded=self.agent.is_unloaded))
        else:
            # Take greedy action
            if self.agent.is_loaded is False and self.agent.is_unloaded is False:
                # Consider q0 table
                q_table = current_state.q0
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is False:
                # Consider q1 table
                q_table = current_state.q1
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is True:
                # Consider q2 table
                q_table = current_state.q2
                return max(q_table, key=q_table.get)

    def __take_action(self, state, action):
        next_state = None
        reward = None
        is_episode_finished = None

        current_state = state
        current_x = current_state.coords[0]
        current_y = current_state.coords[1]

        # If the agent in upper slope
        if current_state.state_type == Constants.upper_slope:
            # If action is right action, it can move to right state with 0.5 probability
            if action == Constants.right_action:
                dice = random.uniform(0, 1)
                if dice < 0.5:
                    # Failure
                    next_state = current_state
                    reward = -0.5
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
                else:
                    # Success
                    next_state = self.__determine_next_state(state=current_state, action=action)
                    reward = self.__determine_reward(state=next_state)
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
            # If action is up action, it will move to upper state
            elif action == Constants.up_action:
                next_state = self.__determine_next_state(state=current_state, action=action)
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished
            # If action is done action, it will stay at the current state
            elif action == Constants.down_action:
                next_state = current_state
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished
        # If the agent in lower slope
        elif current_state.state_type == Constants.lower_slope:
            # If action is left action, it can move to left state with 0.5 probability
            if action == Constants.left_action:
                dice = random.uniform(0, 1)
                if dice < 0.5:
                    # Failure
                    next_state = current_state
                    reward = -0.5
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
                else:
                    # Success
                    next_state = self.__determine_next_state(state=current_state, action=action)
                    reward = self.__determine_reward(state=next_state)
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
            # If action is up action, it will stay at the current state
            elif action == Constants.up_action:
                next_state = current_state
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished
            # If action is done action, it will move to lower state
            elif action == Constants.down_action:
                next_state = self.__determine_next_state(state=current_state, action=action)
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished

        # If the current state is LoadState and agent takes load action, it stays at the current state
        if current_state.state_type == Constants.load_state and action == Constants.load_action:
            next_state = current_state
            reward = self.__determine_reward(state=next_state)
            is_episode_finished = False
            return next_state, reward, is_episode_finished

        # If the current state is UnloadState and agent takes unload action, it stays at the current state
        if current_state.state_type == Constants.unload_state and action == Constants.unload_action:
            next_state = current_state
            reward = self.__determine_reward(state=next_state)
            is_episode_finished = False
            return next_state, reward, is_episode_finished

        next_state = self.__determine_next_state(state=current_state, action=action)

        # If next_state is slope, determine whether it is upper slope or lower slope
        if next_state.state_type == Constants.slope_state:
            # If agent comes that state with either left or down action, means that it is upper slope
            if action == Constants.left_action or action == Constants.down_action:
                next_state = next_state.upper_slope
            # If agent comes that state with either right or up state, means that it is lower slope
            if action == Constants.right_action or action == Constants.up_action:
                next_state = next_state.lower_slope

        reward = self.__determine_reward(state=next_state)

        if reward == 1000:
            is_episode_finished = True
        else:
            is_episode_finished = False

        return next_state, reward, is_episode_finished

    def __determine_next_state(self, state, action):
        if action == Constants.load_action or action == Constants.unload_action:
            return state

        coord_x = state.coords[0]
        coord_y = state.coords[1]

        coord_new_x = None
        coord_new_y = None

        if action == Constants.left_action:
            coord_new_x = coord_x
            coord_new_y = coord_y - 1
        elif action == Constants.right_action:
            coord_new_x = coord_x
            coord_new_y = coord_y + 1
        elif action == Constants.up_action:
            coord_new_x = coord_x - 1
            coord_new_y = coord_y
        elif action == Constants.down_action:
            coord_new_x = coord_x + 1
            coord_new_y = coord_y

        # If new coords are out of bound, return state
        if coord_new_x < 0 or coord_new_x >= self.environment.n:
            return state
        elif coord_new_y < 0 or coord_new_y >= self.environment.n:
            return state

        # If new coords are blocking state, return state
        if self.environment.env[coord_new_x][coord_new_y].state_type == Constants.block_state:
            return state

        return self.environment.env[coord_new_x][coord_new_y]

    def __determine_reward(self, state):
        if state.state_type == Constants.rough_state:
            return -10

        if self.agent.is_loaded is True and self.agent.is_unloaded is True:
            if state.coords == (0, 0):
                return 1000

        return -0.1
