import unittest
from unittest.mock import Mock, patch

from module.exception import GameNotRunningError
from tasks.Component.GeneralBattle.general_battle import GeneralBattle
from tasks.GameUi.assets import GameUiAssets


class GeneralBattleAutoTest(unittest.TestCase):
    def make_task(self):
        task = GeneralBattle.__new__(GeneralBattle)
        task.device = Mock()
        task.device.image = None
        return task

    def test_manual_mode_is_switched_to_auto(self):
        task = self.make_task()
        target = GameUiAssets.O_BATTLE_HAND

        with patch.object(target, 'ocr', return_value=target.keyword), \
                patch.object(target, 'coord', return_value=(42, 64)):
            self.assertTrue(task._ensure_battle_auto())

        task.device.click.assert_called_once_with(
            x=42, y=64, control_name=target.name)

    def test_battle_mode_ocr_is_throttled(self):
        task = self.make_task()
        target = GameUiAssets.O_BATTLE_HAND

        with patch.object(target, 'ocr', return_value='auto') as ocr:
            self.assertFalse(task._ensure_battle_auto())
            self.assertFalse(task._ensure_battle_auto())

        ocr.assert_called_once_with(task.device.image)

    def test_stopped_game_is_reported_after_mode_is_missing(self):
        task = self.make_task()
        task._battle_mode_missing_timer = Mock()
        task._battle_mode_missing_timer.reached.return_value = True
        task.device.app_is_running.return_value = False

        with patch.object(GameUiAssets.O_BATTLE_HAND, 'ocr', return_value=''):
            with self.assertRaisesRegex(GameNotRunningError, 'during battle'):
                task._ensure_battle_auto()

        task.device.app_is_running.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
