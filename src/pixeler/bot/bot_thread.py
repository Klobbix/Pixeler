import ctypes
import platform
import threading


class BotThread(threading.Thread):
    def __init__(self, target: callable):
        threading.Thread.__init__(self)
        self.target = target

    def run(self):
        id = self.__get_id()
        try:
            print(f"Thread started with id {id}")
            self.target()
        finally:
            print(f"Thread {id} stopped")

    def __get_id(self):
        """Returns id of the respective thread"""
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def stop(self):
        """Raises SystemExit exception in the thread. This can be called from the main thread followed by join()."""
        thread_id = self.__get_id()
        if platform.system() == "Windows":
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                print("Exception raise failure")
        elif platform.system() == "Linux":
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
                print("Exception raise failure")
