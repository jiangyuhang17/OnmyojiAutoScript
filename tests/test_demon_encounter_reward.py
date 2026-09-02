import unittest
from types import SimpleNamespace
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

    def test_box_flow_recovers_when_battle_starts(self):
        task = ScriptTask.__new__(ScriptTask)
        task.config = SimpleNamespace(
            demon_encounter=SimpleNamespace(
                box_buy_config=SimpleNamespace(box_buy_sushi=False),
            ),
        )
        task.screenshot = Mock()
        task.appear = Mock(return_value=False)
        task.is_in_prepare = Mock(return_value=True)
        task.is_in_real_battle = Mock(return_value=False)
        task.run_general_battle = Mock(return_value=True)
        task.click = Mock()

        target_click = Mock()
        task._box(target_click)

        task.run_general_battle.assert_called_once_with()
        task.click.assert_not_called()


if __name__ == '__main__':
    unittest.main()
