import threading
import sys
import termios
import tty

ESCAPE = "\x1b"
CTRL_C = "\x03"



class StepController:
    """n = advance | c = continue | q = quit"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._event = threading.Event()
        self._quit = False
        self._advance: int = 0

    def _read_key(self) -> str:
      """Block till a touch is pressed"""
      fd = open("/dev/tty", "rb", buffering=0)
      old_settings = termios.tcgetattr(fd)
      try:
          tty.setraw(fd)
          ch = sys.stdin.read(1)
      finally:
          termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
      if ch == CTRL_C:
          raise KeyboardInterrupt("Ctrl+C")
      return ch.lower()


    def wait(self) -> None:
        
        if self._advance > 0:
          self._advance -= 1
          return
    
        if not self.enabled:
            return

        self._event.clear()

        def _worker():
          try:
            while True:
                key = self._read_key()
                if key == "n":
                    self._event.set()
                    break
                elif key == "c":
                    self.enabled = False
                    self._event.set()
                    break
                elif key == "s":
                    self.enabled = True
                    self._event.set()
                    break
                elif key == "j":
                    self._advance = 5
                    self._event.set()
                    break
                elif key == "q":
                    self._quit = True
                    self._event.set()
                    break
          except Exception as e:
            import traceback
            traceback.print_exc()
            self._event.set()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._event.wait()

    @property
    def should_quit(self) -> bool:
        return self._quit
