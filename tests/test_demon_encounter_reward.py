import unittest
from unittest.mock import Mock

from tasks.DemonEncounter.script_task import ScriptTask


class DemonEncounterRewardTest(unittest.TestCase):
    def test_pending_lantern_reward_is_cleared(self):
        task = ScriptTask.__new__(ScriptTask)
        task.appear = Mock(return_value=True)
        task.ui_click_until_disappear = Mock()

        self.assertTrue(task._clear_pending_battle_reward())
        task.ui_click_until_disappear.assert_called_once_with(task.I_REWARD)

    def test_no_pending_reward_is_ignored(self):
        task = ScriptTask.__new__(ScriptTask)
        task.appear = Mock(return_value=False)
        task.ui_click_until_disappear = Mock()

        self.assertFalse(task._clear_pending_battle_reward())
        task.ui_click_until_disappear.assert_not_called()


if __name__ == '__main__':
    unittest.main()
