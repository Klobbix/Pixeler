import pyautogui as pag


def press(key: str):
    pag.press(key)


def write(text: str, interval: float):
    pag.write(text, interval=interval)


def key_down(key: str):
    pag.keyDown(key)


def key_up(key: str):
    pag.keyUp(key)


def hotkey(key: str):
    pag.hotkey(key)
