import sys
from .store import Store
from .jobs import run

if __name__ == "__main__":
    run(Store(sys.argv[1]), sys.argv[2])
