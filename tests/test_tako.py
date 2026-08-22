import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tasks.Tako.script_task import ScriptTask


class TakoTest(unittest.TestCase):
    def test_disabled_fire_does_not_count_as_teammate(self):
        add = object()
        fire_fail = object()
        timer = Mock()
        task = SimpleNamespace(
            I_ADD_1=add,
            I_FIRE_FAIL=fire_fail,
            appear=Mock(side_effect=lambda image, **kwargs: image is fire_fail),
        )

        result = ScriptTask._teammate_ready(task, timer)

        self.assertFalse(result)
        timer.reset.assert_called_once_with()
        timer.reached.assert_not_called()

    def test_ready_teammate_requires_stable_confirmation(self):
        timer = Mock()
        timer.reached.return_value = True
        task = SimpleNamespace(
            I_ADD_1=object(),
            I_FIRE_FAIL=object(),
            appear=Mock(return_value=False),
        )

        result = ScriptTask._teammate_ready(task, timer)

        self.assertTrue(result)
        timer.reached.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
