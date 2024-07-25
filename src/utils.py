import functools, time, sys

TIMER_ENABLED = True

import functools

def print_function_name(func):
    """Decorator that prints the name of the function whenever it is called."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function '{func.__name__}' is called")
        return func(*args, **kwargs)
    return wrapper





def timer(func):
    """Decorator to time a function and print its name and execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Check if the function is a constructor (__init__) and include the class name
        if TIMER_ENABLED is False:
            return result
        if func.__name__ == '__init__':
            class_name = args[0].__class__.__name__
            print(f"..Constructor for '{class_name}' executed in {end_time - start_time:.4f} seconds.")
        else:
            print(f"..Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        
        return result
    return wrapper


class LogPrint:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass



if __name__ == '__main__':
    pass