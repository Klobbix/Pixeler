# Pixeler
[![Publish Python Package](https://github.com/Klobbix/Pixeler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Klobbix/Pixeler/actions/workflows/python-publish.yml)

### This project is a work-in-progress!

### Human-like Automation and Screen Reading OCR for Python

Pixeler is a Python library that enables developers to create sophisticated automation bots capable of reading and
interacting with a user’s screen in a human-like manner.
By utilizing Optical Character Recognition (OCR) and Bézier curve-based mouse movements, Pixeler provides the tools
necessary for building bots that can perform tasks like identifying text on the screen, responding to UI changes, and
executing precise mouse interactions.


### Features

    OCR-Based Screen Reading: Accurately read and interpret text on the screen, enabling bots to understand what’s happening in real-time.
    Human-like Mouse Movements: Simulate realistic, human-like mouse movements using Bézier curves to interact with UI elements naturally.
    Window-Specific Targeting: Focus on specific windows based on their titles, allowing the bot to interact with the correct application.
    Flexible Automation: Design bots that can respond to changes in the UI dynamically, making them adaptable to various applications.

### Third-Party Setup
1. [Install Tesseract](https://github.com/UB-Mannheim/tesseract/releases/)
2. Add Tesseract install path to PATH environment variables

### Installation

You can install Pixeler via pip:

`pip install pixeler`

# Usage

`import Pixeler.bot`

### Extend the bot

Create your own class extending the Pixeler Bot class:

```
class MyBot(Bot):
    ...
```

### Initialize the bot

```
bot = MyBot(title="Target Application")
```

### Examples
See the **examples** folder. There are plenty of scenarios represented here, from color tracking, screen reading, and Win32 overlays.


### License
Pixeler is licensed under the MIT License. See the LICENSE file for more details.


### Acknowledgments

    pytesseract: For the powerful OCR capabilities.
    OpenCV: For the robust computer vision tools.
    PyAutoGUI: For providing easy-to-use cross-platform GUI automation tools.

### Contact
For any inquiries, please contact [klobbix@gmail.com].