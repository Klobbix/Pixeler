import pyautogui as pag
import pytweening


def position():
    return pag.position()


def move_to(x: int, y: int, duration: float = 0.6, tween=pytweening.easeInOutQuad):
    pag.moveTo(x=x, y=y, duration=duration, tween=tween)


def click():
    pag.click()


def middle_click():
    pag.middleClick()


def right_click():
    pag.rightClick()


def double_click():
    pag.doubleClick()
