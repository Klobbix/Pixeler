import time
from abc import ABC, abstractmethod

from src.bot.bot_status import BotStatus
from src.bot.bot_thread import BotThread
from src.input.window import Window


class Bot(ABC):
    def __init__(self, window: Window):
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
        if self.status == BotStatus.STOPPED:
            self.log("Starting bot...")
            self.status = BotStatus.RUNNING
            self.on_start()
            self.thread = BotThread(target=self.loop)
            self.thread.daemon = True
            self.thread.start()
            self.start_time = time.time()
        elif self.status == BotStatus.RUNNING:
            self.log("Bot is already running.")

    def stop(self):
        if self.status == BotStatus.STOPPED:
            self.log("Bot is not running.")
        elif self.status == BotStatus.RUNNING:
            self.log("Stopping bot...")
            self.status = BotStatus.STOPPED
            self.thread.stop()
            self.thread.join()
            self.on_stop()

    def log(self, message: str):
        message = f"{time.ctime()}: {message}"
        print(message)
