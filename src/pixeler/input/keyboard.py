import pyautogui as pag


def press(key: str):
    """
    Types a key.
    :param key: The key to type
    :return:
    """
    pag.press(key)


def write(text: str, interval: float = 0.1):
    """
    Writes text out where each letter takes a specified interval
    :param text: The text to write
    :param interval: The interval between each letter
    :return:
    """
    pag.write(text, interval=interval)


def key_down(key: str):
    """
    Performs a keydown action.
    :param key: The key to 'key down'
    :return:
    """
    pag.keyDown(key)


def key_up(key: str):
    """
    Performs a keyup action.
    :param key: The key to 'key up'
    :return:
   """
    pag.keyUp(key)


def hotkey(key: str):
    """
    Performs a hotkey action.
    :param key: The key to 'hotkey'
    :return:
    """
    pag.hotkey(key)
