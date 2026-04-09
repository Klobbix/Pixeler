"""
Transparent Win32 overlay window for bot visualisation.

The overlay sits topmost and click-through over a target game window,
letting you see exactly what the bot detects and where it intends to act.

Usage::

    window = Win32Window("My Game")
    overlay = window.create_overlay()   # starts message loop automatically

    # Inside bot.step():
    overlay.begin_frame()               # clear previous frame, sync position
    overlay.draw_rect((10, 20), (80, 60), RED)
    overlay.draw_text("HP: 95", 10, 10, WHITE)
    overlay.draw_circle((400, 300), 8, GREEN)
    overlay.end_frame()

    # On shutdown:
    overlay.stop()

Coordinate system
-----------------
All drawing coordinates are **relative to the game window's client area top-left**,
which aligns with pixel positions returned by Win32Window.screenshot() and the
vision module.  Both the screenshot and the overlay use ClientToScreen +
GetClientRect so that title-bar height, window borders, and the invisible DWM
resize frame are all excluded from the coordinate space.
"""

import ctypes
import threading

import win32api
import win32con
import win32gui

from pixeler.vision.color import Color

# GDI line-drawing functions not wrapped by win32gui — load via ctypes at module level
_MoveToEx = getattr(ctypes.windll.gdi32, "MoveToEx")
_LineTo = getattr(ctypes.windll.gdi32, "LineTo")

# Transparent background colour — must match SetLayeredWindowAttributes key
_COLORKEY = 0x000000  # black = transparent


class Overlay:
    """Transparent, click-through, always-on-top overlay for a parent HWND."""

    _CLASS_NAME = "PixelerOverlay"

    def __init__(self, parent_hwnd: int):
        self.parent_hwnd = parent_hwnd
        self.hwnd: int | None = None
        self._hdc: int | None = None

        # Pen/brush caches keyed by (b, g, r, thickness) or (b, g, r)
        self._pens: dict[tuple, int] = {}
        self._brushes: dict[tuple, int] = {}

        self._msg_thread: threading.Thread | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 3.0) -> None:
        """
        Create the overlay window on a background daemon thread and block
        until it is ready to receive drawing calls.

        :raises RuntimeError: If the window fails to create within *timeout* seconds.
        """
        t = threading.Thread(
            target=self._message_loop,
            daemon=True,
            name="PixelerOverlay",
        )
        self._msg_thread = t
        t.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("Overlay window did not initialise within timeout")

    def stop(self) -> None:
        """Destroy the overlay and release all GDI resources."""
        if self.hwnd:
            win32gui.PostMessage(self.hwnd, win32con.WM_DESTROY, 0, 0)
        if self._msg_thread:
            self._msg_thread.join(timeout=2.0)
        self._msg_thread = None
        self.hwnd = None

    # ------------------------------------------------------------------
    # Message loop (runs on its own daemon thread)
    # ------------------------------------------------------------------

    def _message_loop(self) -> None:
        self._create_window()
        self._ready.set()
        win32gui.PumpMessages()

    def _create_window(self) -> None:
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = self._CLASS_NAME
        wc.lpfnWndProc = self._wnd_proc
        wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wc.hbrBackground = 0  # Prevent system from erasing background

        try:
            win32gui.RegisterClass(wc)
        except OSError:
            pass  # Already registered from a previous run in the same process

        x, y = win32gui.ClientToScreen(self.parent_hwnd, (0, 0))
        _, _, w, h = win32gui.GetClientRect(self.parent_hwnd)

        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
            self._CLASS_NAME,
            "PixelerOverlay",
            win32con.WS_POPUP,
            x, y, w, h,
            self.parent_hwnd, None, wc.hInstance, None,
        )

        # Black pixels become transparent; alpha channel is unused
        win32gui.SetLayeredWindowAttributes(
            self.hwnd, _COLORKEY, 0, win32con.LWA_COLORKEY
        )
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)

    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == win32con.WM_DESTROY:
            self._release_resources()
            win32gui.PostQuitMessage(0)
            return 0
        # WM_PAINT: validate the region so Windows stops sending paint messages.
        # We draw directly to the cached DC outside the paint cycle, so there
        # is nothing to redraw here.
        if msg == win32con.WM_PAINT:
            hdc, ps = win32gui.BeginPaint(hwnd)
            win32gui.EndPaint(hwnd, ps)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def _get_hdc(self) -> int:
        if self._hdc is None:
            self._hdc = win32gui.GetDC(self.hwnd)
        hdc = self._hdc
        assert hdc is not None
        return hdc

    def _get_pen(self, color: Color, thickness: int = 1) -> int:
        bgr = (int(color.lower[0]), int(color.lower[1]), int(color.lower[2]))
        key = (*bgr, thickness)
        if key not in self._pens:
            self._pens[key] = win32gui.CreatePen(
                win32con.PS_SOLID, thickness, win32api.RGB(bgr[2], bgr[1], bgr[0])
            )
        return self._pens[key]

    def _get_brush(self, color: Color) -> int:
        bgr = (int(color.lower[0]), int(color.lower[1]), int(color.lower[2]))
        if bgr not in self._brushes:
            self._brushes[bgr] = win32gui.CreateSolidBrush(win32api.RGB(bgr[2], bgr[1], bgr[0]))
        return self._brushes[bgr]

    def _release_resources(self) -> None:
        for handle in self._pens.values():
            win32gui.DeleteObject(handle)
        for handle in self._brushes.values():
            win32gui.DeleteObject(handle)
        self._pens.clear()
        self._brushes.clear()
        if self._hdc is not None:
            win32gui.ReleaseDC(self.hwnd, self._hdc)
            self._hdc = None

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Erase all drawn content from the overlay."""
        if not self.hwnd:
            return
        hdc = self._get_hdc()
        client = win32gui.GetClientRect(self.hwnd)
        black_brush = win32gui.GetStockObject(win32con.BLACK_BRUSH)
        win32gui.FillRect(hdc, client, black_brush)
        # BLACK_BRUSH is a stock object — do not DeleteObject it

    def begin_frame(self) -> None:
        """
        Start a new frame: sync the overlay position to the parent window
        and clear all previously drawn content.

        Call this at the start of every bot step() before any drawing.
        """
        self.sync_to_parent()
        self.clear()

    def end_frame(self) -> None:
        """
        Flush the current frame.

        GDI drawing to a DC is immediate, so this is a lightweight call
        that simply ensures the window is up-to-date.
        """
        if self.hwnd:
            win32gui.UpdateWindow(self.hwnd)

    def sync_to_parent(self) -> None:
        """Reposition and resize the overlay to exactly cover the parent window's client area."""
        if not self.hwnd:
            return
        x, y = win32gui.ClientToScreen(self.parent_hwnd, (0, 0))
        _, _, w, h = win32gui.GetClientRect(self.parent_hwnd)
        win32gui.SetWindowPos(
            self.hwnd,
            win32con.HWND_TOPMOST,
            x, y, w, h,
            win32con.SWP_NOACTIVATE,
        )

    # ------------------------------------------------------------------
    # Drawing primitives
    # All coordinates are in client space (relative to the game window's
    # top-left corner, matching screenshot pixel coordinates).
    # ------------------------------------------------------------------

    def draw_rect(self,
                  top_left: tuple[int, int],
                  bottom_right: tuple[int, int],
                  color: Color,
                  thickness: int = 1,
                  filled: bool = False) -> None:
        """
        Draw a rectangle outline or filled rectangle.

        :param top_left:     (x, y) of the top-left corner.
        :param bottom_right: (x, y) of the bottom-right corner.
        :param color:        Stroke and fill colour.
        :param thickness:    Outline thickness in pixels (ignored when filled).
        :param filled:       Fill the interior when True.
        """
        hdc = self._get_hdc()
        pen = self._get_pen(color, thickness)
        brush = (self._get_brush(color) if filled
                 else win32gui.GetStockObject(win32con.NULL_BRUSH))
        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, brush)
        win32gui.Rectangle(hdc,
                            top_left[0], top_left[1],
                            bottom_right[0], bottom_right[1])

    def draw_circle(self,
                    center: tuple[int, int],
                    radius: int,
                    color: Color,
                    thickness: int = 1,
                    filled: bool = False) -> None:
        """
        Draw a circle centred on *center*.

        :param center: (x, y) of the centre.
        :param radius: Radius in pixels.
        """
        hdc = self._get_hdc()
        pen = self._get_pen(color, thickness)
        brush = (self._get_brush(color) if filled
                 else win32gui.GetStockObject(win32con.NULL_BRUSH))
        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, brush)
        cx, cy = center
        win32gui.Ellipse(hdc, cx - radius, cy - radius, cx + radius, cy + radius)

    def draw_line(self,
                  start: tuple[int, int],
                  end: tuple[int, int],
                  color: Color,
                  thickness: int = 1) -> None:
        """Draw a straight line between *start* and *end*."""
        hdc = self._get_hdc()
        pen = self._get_pen(color, thickness)
        win32gui.SelectObject(hdc, pen)
        _MoveToEx(hdc, start[0], start[1], None)
        _LineTo(hdc, end[0], end[1])

    def draw_crosshair(self,
                       center: tuple[int, int],
                       size: int,
                       color: Color,
                       thickness: int = 1) -> None:
        """
        Draw a crosshair (cross) centred on *center*.

        Useful for marking the bot's current target point.
        """
        cx, cy = center
        self.draw_line((cx - size, cy), (cx + size, cy), color, thickness)
        self.draw_line((cx, cy - size), (cx, cy + size), color, thickness)

    def draw_text(self,
                  text: str,
                  x: int,
                  y: int,
                  color: Color,
                  flags: int = win32con.DT_LEFT | win32con.DT_TOP) -> None:
        """
        Draw a text string at position (x, y).

        :param x:     Left edge of the text in client coordinates.
        :param y:     Top edge of the text in client coordinates.
        :param flags: win32con.DT_* flags controlling text layout.
        """
        hdc = self._get_hdc()
        bgr = (int(color.lower[0]), int(color.lower[1]), int(color.lower[2]))
        win32gui.SetTextColor(hdc, win32api.RGB(bgr[2], bgr[1], bgr[0]))
        win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
        client = win32gui.GetClientRect(self.hwnd)
        rect = (x, y, client[2], client[3])
        win32gui.DrawText(hdc, text, -1, rect, flags)

    def draw_label(self,
                   text: str,
                   top_left: tuple[int, int],
                   text_color: Color,
                   bg_color: Color | None = None) -> None:
        """
        Draw a text label, optionally with a filled background rect for
        legibility over busy game visuals.

        :param top_left:   Position of the label's top-left corner.
        :param bg_color:   Background fill colour. No background if None.
        """
        x, y = top_left
        # Approximate character width/height for background sizing
        char_w, char_h = 7, 14
        if bg_color is not None:
            w = len(text) * char_w + 4
            self.draw_rect((x - 2, y - 1), (x + w, y + char_h + 1),
                           bg_color, filled=True)
        self.draw_text(text, x, y, text_color)
