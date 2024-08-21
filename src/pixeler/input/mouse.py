import pyautogui as pag
import pytweening


def position():
    """
    Returns the mouse position
    :return: The mouse position
    """
    return pag.position()


def move_to(x: int, y: int, duration: float = 0.6, tween=pytweening.easeInOutQuad):
    """
    Moves the mouse to the specified position.
    :param x: The x position
    :param y: The y position
    :param duration: The duration of the movement
    :param tween: The tweening function
    :return:
    """
    pag.moveTo(x=x, y=y, duration=duration, tween=tween)


def click():
    """
    Left-clicks the mouse button
    :return:
    """
    pag.click()


def middle_click():
    """
    Middle-clicks the mouse button
    :return:
    """
    pag.middleClick()


def right_click():
    """
    Right-clicks the mouse button
    :return:
    """
    pag.rightClick()


def double_click():
    """
    Double-clicks the left mouse button
    :return:
    """
    pag.doubleClick()
