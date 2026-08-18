import unittest
from unittest.mock import Mock

from tasks.Component.SwitchSoul.switch_soul import SwitchSoul


class SwitchSoulTest(unittest.TestCase):
    def make_task(self):
        task = SwitchSoul.__new__(SwitchSoul)
        task.screenshot = Mock()
        return task

    def test_confirmation_stops_after_dialog_disappears(self):
        task = self.make_task()
        task.appear = Mock(side_effect=[True, False])
        task.appear_then_click = Mock(side_effect=[False, True, False])

        self.assertTrue(task._wait_and_confirm_soul_switch(timeout=0))
        task.appear_then_click.assert_any_call(
            task.I_SOU_SWITCH_SURE, interval=0.8
        )

    def test_successful_switch_does_not_click_team_again(self):
        task = self.make_task()
        target_team = Mock()
        task.appear = Mock(return_value=False)
        task.appear_then_click = Mock(return_value=True)
        task._wait_and_confirm_soul_switch = Mock(return_value=True)

        self.assertTrue(task._switch_soul_team(target_team))
        task.appear_then_click.assert_called_once_with(target_team, interval=0.8)
        task._wait_and_confirm_soul_switch.assert_called_once_with()

    def test_missing_dialog_reports_failure(self):
        task = self.make_task()
        task.appear = Mock(return_value=False)
        task.appear_then_click = Mock(return_value=False)
        task._wait_and_confirm_soul_switch = Mock(return_value=False)

        self.assertFalse(task._switch_soul_team(Mock(), attempts=3))
        self.assertEqual(task.appear_then_click.call_count, 3)


if __name__ == '__main__':
    unittest.main()
