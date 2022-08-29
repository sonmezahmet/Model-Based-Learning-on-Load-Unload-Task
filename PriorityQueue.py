class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)

    # for popping an element based on Priority
    def pop(self):
        try:
            max_val = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[max_val][0]:
                    max_val = i
            state = self.queue[max_val][1]
            action = self.queue[max_val][2]
            is_loaded = self.queue[max_val][3]
            is_unloaded = self.queue[max_val][4]
            del self.queue[max_val]
            return state, action, is_loaded, is_unloaded
        except IndexError:
            print()
            exit()
