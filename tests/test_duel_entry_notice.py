import unittest
from unittest.mock import Mock

from tasks.Duel.script_task import ScriptTask


class DuelEntryNoticeTest(unittest.TestCase):
    def test_dismisses_season_trial_notice(self):
        task = ScriptTask.__new__(ScriptTask)
        task.appear_then_click = Mock(return_value=True)

        self.assertTrue(task.dismiss_duel_entry_notice())
        task.appear_then_click.assert_called_once_with(
            task.I_SEASON_TRIAL_NOTICE, interval=1
        )

    def test_ignores_normal_duel_page(self):
        task = ScriptTask.__new__(ScriptTask)
        task.appear_then_click = Mock(return_value=False)

        self.assertFalse(task.dismiss_duel_entry_notice())


if __name__ == '__main__':
    unittest.main()
