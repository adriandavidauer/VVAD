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
        print(f"data is {logtime_data}")
        print("name is", sleep_x_sec.__name__)
        self.assertEqual(logtime_data.get("SLEEP_X_SEC"), sleep_time * 1000)


if __name__ == '__main__':
    unittest.main()
