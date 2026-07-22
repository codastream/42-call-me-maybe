import threading
import termios
import tty

ESCAPE = "\x1b"
CTRL_C = "\x03"


class StepController:
    """Manage step-by-step execution"""

    def __init__(self, enabled: bool = True):
        """Initializer"""
        self.enabled = enabled
        self._event = threading.Event()
        self._quit = False
        self._advance: int = 0

    def _read_key(self) -> str:
        """Read key from /dev/tty"""
        with open("/dev/tty", "rb", buffering=0) as fd:
            fileno = fd.fileno()
            old_settings = termios.tcgetattr(fileno)
            try:
                tty.setraw(fileno)
                ch = fd.read(1).decode(errors="replace")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if ch == CTRL_C:
            raise KeyboardInterrupt("Ctrl+C")
        return ch.lower()

    def wait(self) -> None:
        """Block execution till a command is pressed"""

        if self._advance > 0:
            self._advance -= 1
            return

        if not self.enabled:
            return

        self._event.clear()

        def _worker() -> None:
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
                    elif key == "j":
                        self._advance = 5
                        self._event.set()
                        break
                    elif key == "q":
                        self._quit = True
                        self._event.set()
                        break
            except Exception:
                import traceback
                traceback.print_exc()
                self._event.set()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._event.wait()

    @property
    def should_quit(self) -> bool:
        """Return True if q has been pressed"""
        return self._quit
