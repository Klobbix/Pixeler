import time
from abc import ABC, abstractmethod
from typing import Union

from src.pixeler.bot.bot_status import BotStatus
from src.pixeler.bot.bot_thread import BotThread
from src.pixeler.input.win32_window import Win32Window
from src.pixeler.input.window import Window


class Bot(ABC):
    def __init__(self, window: Union[Window | Win32Window] = None):
        self.window = window
        self.status = BotStatus.STOPPED
        self.thread: BotThread = None
        self.start_time = 0.0

    @abstractmethod
    def on_start(self):
        """
        Called before the loop thread is started.
        :return:
        """
        pass

    @abstractmethod
    def on_stop(self):
        """
        Called after the loop thread is stopped.
        :return:
        """
        pass

    @abstractmethod
    def loop(self):
        """
        Main logic of the bot. This function is called in the BotThread and should be a while loop.
        """
        pass

    def start(self):
        """
        Starts the bot by updating the internal status to RUNNING and starting the loop function on the BotThread.
        :return:
        """
        if self.status == BotStatus.STOPPED:
            self.log("Starting bot...")
            self.status = BotStatus.RUNNING
            self.start_time = time.time()
            self.on_start()
            self.thread = BotThread(target=self.loop)
            self.thread.start()
        elif self.status == BotStatus.RUNNING:
            self.log("Bot is already running.")

    def stop(self):
        """
        Stops the bot, updates the internal status to STOPPED, and stops/joins the BotThread.
        :return:
        """
        if self.status == BotStatus.STOPPED:
            self.log("Bot is not running.")
        elif self.status == BotStatus.RUNNING:
            self.log("Stopping bot...")
            self.status = BotStatus.STOPPED
            self.log(f"Ran for {time.time() - self.start_time} seconds.")
            self.thread.stop()
            self.thread.join()
            self.window.__destroy()
            self.on_stop()

    def log(self, message: str):
        """
        Generic log function.
        :param message: The message to log.
        :return:
        """
        message = f"{time.ctime()}: {message}"
        print(message)
