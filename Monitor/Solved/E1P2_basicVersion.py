#! /usr/bin/python3

import time
import random
import sys
from multiprocessing import Process, Lock, Condition, Value, Array



### Monitor start

"""
class Buffer:
    def __init__(self, nb_cases):
        self.nb_cases = nb_cases
        self.storage_val = [-1] * nb_cases
        self.storage_type = [-1] * nb_cases
        self.ptr_prod = 0
        self.ptr_cons = 0

    def produce(self, msg_val, msg_type, msg_source):
        position = self.ptr_prod
        print('process %2d starts prod %2d (type:%d) in place %2d and the buffer is\t\t %s' %
              (msg_source, msg_val, msg_type, position, self.storage_val[:]))
        time.sleep(random.random())
        self.storage_val[position] = msg_val
        self.storage_type[position] = msg_type
        self.ptr_prod = (position + 1) % self.nb_cases
        print('process %2d    produced %2d (type:%d) in place %2d and the buffer is\t\t %s' %
              (msg_source, msg_val, msg_type, position, self.storage_val[:]))


    def consume(self, id_cons):
        position = self.ptr_cons
        result = self.storage_val[position]
        result_type = self.storage_type[position]
        print('\tprocess %2d starts cons %2d (type:%d) in place %2d and the buffer is\t %s' %
              (id_cons, result, result_type, position, self.storage_val[:]))
        time.sleep(random.random())
        self.storage_val[position] = -1
        self.storage_type[position] = -1
        self.ptr_cons = (position + 1) % self.nb_cases
        print('\tprocess %2d    consumed %2d (type:%d) in place %2d and the buffer is\t %s' %
              (id_cons, result, result_type, position, self.storage_val[:]))
        return result
"""
"""
class Buffer:
    def __init__(self, nb_cases):
        self.nb_cases = nb_cases
        self.storage_val = Array('i', [-1] * nb_cases)  # Shared array for buffer values
        self.storage_type = Array('i', [-1] * nb_cases)  # Shared array for message types
        self.ptr_prod = Value('i', 0)  # Shared pointer for producers
        self.ptr_cons = Value('i', 0)  # Shared pointer for consumers

    def produce(self, msg_val, msg_type, msg_source):
        with self.ptr_prod.get_lock():  # Ensure atomic operations on the shared pointer
            position = self.ptr_prod.value
            print('process %2d starts prod %2d (type:%d) in place %2d and the buffer is\t\t %s' %
                  (msg_source, msg_val, msg_type, position, list(self.storage_val[:])))
            time.sleep(random.random())
            self.storage_val[position] = msg_val
            self.storage_type[position] = msg_type
            self.ptr_prod.value = (position + 1) % self.nb_cases
            print('process %2d    produced %2d (type:%d) in place %2d and the buffer is\t\t %s' %
                  (msg_source, msg_val, msg_type, position, list(self.storage_val[:])))

    def consume(self, id_cons):
        with self.ptr_cons.get_lock():  # Ensure atomic operations on the shared pointer
            position = self.ptr_cons.value
            result = self.storage_val[position]
            result_type = self.storage_type[position]
            print('\tprocess %2d starts cons %2d (type:%d) in place %2d and the buffer is\t %s' %
                  (id_cons, result, result_type, position, list(self.storage_val[:])))
            time.sleep(random.random())
            self.storage_val[position] = -1
            self.storage_type[position] = -1
            self.ptr_cons.value = (position + 1) % self.nb_cases
            print('\tprocess %2d    consumed %2d (type:%d) in place %2d and the buffer is\t %s' %
                  (id_cons, result, result_type, position, list(self.storage_val[:])))
            return result
"""
class Buffer:
    def __init__(self, nb_cases):
        self.nb_cases = nb_cases
        self.storage_val = Array('i', [-1] * nb_cases)  # Shared buffer for values
        self.storage_type = Array('i', [-1] * nb_cases)  # Shared buffer for types
        self.ptr_prod = Value('i', 0)  # Shared producer pointer
        self.ptr_cons = Value('i', 0)  # Shared consumer pointer
        self.count = Value('i', 0)  # Count of items in the buffer

        # Synchronization tools
        self.lock = Lock()
        self.not_full = Condition(self.lock)  # Condition for full buffer
        self.not_empty = Condition(self.lock)  # Condition for empty buffer

    def produce(self, msg_val, msg_type, msg_source):
        with self.lock:  # Acquire lock for mutual exclusion
            while self.count.value == self.nb_cases:  # Wait if the buffer is full
                self.not_full.wait()

            position = self.ptr_prod.value
            print(f'Producer {msg_source} starts producing {msg_val} (type: {msg_type}) in position {position}')
            self.storage_val[position] = msg_val
            self.storage_type[position] = msg_type
            self.ptr_prod.value = (position + 1) % self.nb_cases
            self.count.value += 1
            print(f'Producer {msg_source} produced {msg_val} (type: {msg_type}) in position {position}')

            self.not_empty.notify()  # Notify a waiting consumer

    def consume(self, id_cons):
        with self.lock:  # Acquire lock for mutual exclusion
            while self.count.value == 0:  # Wait if the buffer is empty
                self.not_empty.wait()

            position = self.ptr_cons.value
            msg_val = self.storage_val[position]
            msg_type = self.storage_type[position]
            print(f'Consumer {id_cons} starts consuming {msg_val} (type: {msg_type}) from position {position}')
            self.storage_val[position] = -1
            self.storage_type[position] = -1
            self.ptr_cons.value = (position + 1) % self.nb_cases
            self.count.value -= 1
            print(f'Consumer {id_cons} consumed {msg_val} (type: {msg_type}) from position {position}')

            self.not_full.notify()  # Notify a waiting producer
            return msg_val


#### Monitor end

def producer(msg_val, msg_type, msg_source, nb_times, buffer):
    for _ in range(nb_times):
        time.sleep(random.random())
        buffer.produce(msg_val, msg_type, msg_source)
        msg_val += 1


def consumer(id_cons, nb_times, buffer):
    for _ in range(nb_times):
        time.sleep(random.random())
        buffer.consume(id_cons)



if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: %s <Nb Prod <= 20> <Nb Conso <= 20> <Nb Cases <= 20> <Nb Times Prod> <Nb Times Cons>" % sys.argv[0])
        sys.exit(1)

    nb_prod = int(sys.argv[1])
    nb_cons = int(sys.argv[2])
    nb_cases = int(sys.argv[3])
    nb_times_prod = int(sys.argv[4])  # Runtime parameter for number of producer iterations
    nb_times_cons = int(sys.argv[5])  # Runtime parameter for number of consumer iterations


    buffer = Buffer(nb_cases)
    
    producers, consumers = [], []
    
    for id_prod in range(nb_prod):
        msg_val_start, msg_type, msg_source = id_prod * nb_times_prod, id_prod % 2, id_prod
        prod = Process(target=producer, args=(msg_val_start, msg_type, msg_source, nb_times_prod, buffer))
        prod.start()
        producers.append(prod)

    for id_cons in range(nb_cons):
        cons=Process(target=consumer, args=(id_cons, nb_times_cons, buffer))
        cons.start()
        consumers.append(cons)

    for prod in producers:
        prod.join()

    for cons in consumers:
        cons.join()