import multiprocessing
import os
import time


def coding(num):
    for i in range(0, num):
        print("coding pid : %d  ppid : %d" % (os.getpid(), os.getppid()))
        time.sleep(0.1)


def music(num):
    for i in range(0, num):
        print("music pid : %d  ppid : %d" % (os.getpid(), os.getppid()))
        time.sleep(0.1)


if __name__ == '__main__':
    codingProcess = multiprocessing.Process(target=coding, kwargs={"num": 1})
    musicProcess = multiprocessing.Process(target=music, args=(2,))
    codingProcess.start()
    musicProcess.start()
    print("pid : %d" % os.getpid())
