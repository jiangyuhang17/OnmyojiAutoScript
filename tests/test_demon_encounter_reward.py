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

    def test_find_boss_reselects_after_center_click(self):
        class BossMapRefreshed(Exception):
            pass

        class DisabledBestBossConfig:
            def __getattr__(self, name):
                return False

        task = ScriptTask.__new__(ScriptTask)
        task.conf = SimpleNamespace(best_demon_boss_config=DisabledBestBossConfig())
        task.device = Mock()
        task.screenshot = Mock()
        task._clear_pending_battle_reward = Mock(return_value=False)

        boss_select_checks = 0
        center_clicks = 0

        def appear(target):
            nonlocal boss_select_checks
            if target is task.I_DE_BOSS:
                boss_select_checks += 1
                if boss_select_checks == 2:
                    raise BossMapRefreshed
                return True
            return False

        def click(target, interval):
            nonlocal center_clicks
            if target is task.C_DM_BOSS_CLICK:
                center_clicks += 1
                self.assertEqual(center_clicks, 1)
            return True

        task.appear = appear
        task.click = click

        with self.assertRaises(BossMapRefreshed):
            task.execute_boss()

        self.assertEqual(boss_select_checks, 2)
        self.assertEqual(center_clicks, 1)


if __name__ == '__main__':
    unittest.main()
