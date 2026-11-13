
import time


class GestureDetector:
    def __init__(self):
        # Eye thresholds
        self.BLINK_CLOSED = 0.50
        self.EYE_OPEN = 0.40

        # Gesture timing (milliseconds)
        self.WINK_HOLD = 250
        self.TILT_HOLD = 450
        self.COOLDOWN = 900

        # Head tilt sensitivity
        self.TILT_THRESHOLD = 15
        self.NEUTRAL_THRESHOLD = 8

        # Change to 1 if the tilt directions are reversed
        self.LEFT_ROLL_SIGN = -1

        self.neutral_roll = None
        self.roll_samples = []

        self.gestures = {
            "left_wink": {"start": None, "fired": False, "last": 0},
            "right_wink": {"start": None, "fired": False, "last": 0},
            "left_tilt": {"start": None, "fired": False, "last": 0},
            "right_tilt": {"start": None, "fired": False, "last": 0},
        }

    def _check_gesture(self, name, active, hold_time, now):
        state = self.gestures[name]

        if not active:
            state["start"] = None
            state["fired"] = False
            return False

        if state["fired"]:
            return False

        if state["start"] is None:
            state["start"] = now
            return False

        held = (now - state["start"]) * 1000

        if held >= hold_time:
            if (now - state["last"]) * 1000 >= self.COOLDOWN:
                state["last"] = now
                state["fired"] = True
                return True

        return False

    def update(self, left_blink, right_blink, roll):
        now = time.monotonic()
        events = []

        # Calibrate neutral head position using first 60 frames.
        if self.neutral_roll is None:
            self.roll_samples.append(roll)

            if len(self.roll_samples) >= 60:
                self.neutral_roll = sum(self.roll_samples) / len(self.roll_samples)

            return ["CALIBRATING HEAD POSITION"]

        # Detect deliberate asymmetric winks.
        left_wink = (
            left_blink > self.BLINK_CLOSED
            and right_blink < self.EYE_OPEN
        )

        right_wink = (
            right_blink > self.BLINK_CLOSED
            and left_blink < self.EYE_OPEN
        )

        if self._check_gesture(
            "left_wink", left_wink, self.WINK_HOLD, now
        ):
            events.append("LEFT_CLICK")

        if self._check_gesture(
            "right_wink", right_wink, self.WINK_HOLD, now
        ):
            events.append("RIGHT_CLICK")

        # Detect head tilts relative to calibrated neutral position.
        relative_roll = (roll - self.neutral_roll) * self.LEFT_ROLL_SIGN

        left_tilt = relative_roll > self.TILT_THRESHOLD
        right_tilt = relative_roll < -self.TILT_THRESHOLD

        if self._check_gesture(
            "left_tilt", left_tilt, self.TILT_HOLD, now
        ):
            events.append("UNDO")

        if self._check_gesture(
            "right_tilt", right_tilt, self.TILT_HOLD, now
        ):
            events.append("REDO")

        return events
