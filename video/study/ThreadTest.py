import threading
import time


def getInfo():
    time.sleep(0.1)
    threading_info = threading.current_thread()
    print(threading_info)


def coding(num):
    for i in range(0, num):
        print("coding")
        time.sleep(0.1)


def music(num):
    for i in range(0, num):
        print("music")
        time.sleep(0.1)


if __name__ == '__main__':
    coding_thread = threading.Thread(target=coding, args=(3,))
    music_thread = threading.Thread(target=music, kwargs={"num": 2})
    coding_thread.start()
    music_thread.start()
    for i in range(10):
        info = threading.Thread(target=getInfo())
        info.start()
