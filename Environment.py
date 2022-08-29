from States.State import State
from States.RegularState import RegularState
from States.LoadState import LoadState
from States.UnloadState import UnloadState
from States.SlopeState import SlopeState
from States.BlockState import BlockState
from States.RoughState import RoughState
import Constants


class Environment:
    def __init__(self, n):
        if n >= 6:
            self.n = n
        else:
            print(f'Minimum size of environment is 6. Instead of {n}x{n}, 6x6 environment has been creating.')
            self.n = 6

        # Create environment
        self.env = self.__create_environment()

    def print_environment(self):
        environment = []

        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i == 0 and j == 0:
                    row.append('S')
                elif self.env[i][j].state_type == Constants.block_state:
                    row.append('B')
                elif self.env[i][j].state_type == Constants.load_state:
                    row.append('L')
                elif self.env[i][j].state_type == Constants.regular_state:
                    row.append('0')
                elif self.env[i][j].state_type == Constants.rough_state:
                    row.append('R')
                elif self.env[i][j].state_type == Constants.slope_state:
                    row.append('\\')
                elif self.env[i][j].state_type == Constants.unload_state:
                    row.append('U')

            environment.append(row)

        # Print
        for i in range(self.n):
            row = ''
            for j in range(self.n):
                row += f'{environment[i][j]:<4}'
            print(row)

    def __create_environment(self):
        raw_environment = self.__create_2d_matrix()

        # Place Start, Load and Unload states
        raw_environment[0][0] = RegularState(coords=(0, 0))
        raw_environment[self.n - 1][self.n - 1] = LoadState(coords=(self.n - 1, self.n - 1))
        raw_environment[0][self.n - 1] = UnloadState(coords=(0, self.n - 1))

        # Place slope boundaries at bottom left corner
        boundary_index = [5, 0]
        offset = self.n - 5
        for i in range(offset):
            raw_environment[boundary_index[0] + i][boundary_index[1] + i] = \
                SlopeState(coords=(boundary_index[0] + i, boundary_index[1] + i))

        # Place slope boundaries at middle
        boundary_index = [3, 1]
        offset = self.n - 4
        for i in range(offset):
            raw_environment[boundary_index[0] + i][boundary_index[1] + i] = \
                SlopeState(coords=(boundary_index[0] + i, boundary_index[1] + i))

        # Place boundaries at upper right corner
        raw_environment[2][3] = SlopeState(coords=(2, 3))
        boundary_index = [1, 3]
        offset = self.n - 4
        for i in range(offset):
            raw_environment[boundary_index[0]][boundary_index[1] + i] = \
                BlockState(coords=(boundary_index[0], boundary_index[1] + i))
            raw_environment[boundary_index[0] - 1][boundary_index[1] + i] = \
                RoughState(coords=(boundary_index[0] - 1, boundary_index[1] + i))

        boundary_index = [2, 4]
        offset = self.n - 5
        for i in range(offset):
            raw_environment[boundary_index[0]][boundary_index[1] + i] = \
                BlockState(coords=(boundary_index[0], boundary_index[1] + i))

        # Traverse raw environment and replace empty_states states with regular states
        for i in range(self.n):
            for j in range(self.n):
                if raw_environment[i][j].state_type == Constants.empty_state:
                    raw_environment[i][j] = RegularState(coords=(i, j))

        print('Environment is created.')

        return raw_environment

    def __create_2d_matrix(self):
        output = []

        for i in range(self.n):
            row = []
            for j in range(self.n):
                row.append(State(state_type=Constants.empty_state, coords=(i, j)))
            output.append(row)

        return output
