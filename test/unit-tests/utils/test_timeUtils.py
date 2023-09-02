import time
import unittest

import vvadlrs3.utils.timeUtils as tUtils


@tUtils.timeit
def sleep_x_sec(sleep_time, **kwargs):
    time.sleep(sleep_time)


class TestTimeUtils(unittest.TestCase):
    """
        Test the time measurement using a sleep function
    """

    def test_timeit(self):
        logtime_data = {}
        sleep_time = 5
        sleep_x_sec(sleep_time, log_time=logtime_data)

        # error message in case if test case got failed
        message = "first value is not less than or equal to 25ms."

        difference = logtime_data.get("SLEEP_X_SEC") - sleep_time * 1000
        self.assertLessEqual(difference, 25, message)


if __name__ == '__main__':
    unittest.main()
