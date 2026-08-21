import unittest
from unittest.mock import Mock

from tasks.GameUi.page import page_exploration
from tasks.WantedQuests.script_task import ScriptTask


class WantedQuestsChallengeTest(unittest.TestCase):
    def test_missing_challenge_panel_returns_to_exploration(self):
        task = ScriptTask.__new__(ScriptTask)
        goto = Mock()
        task.ui_click = Mock(return_value=False)
        task.ui_get_current_page = Mock()
        task.ui_goto = Mock()

        self.assertFalse(task.challenge(goto, 1))
        task.ui_click.assert_called_once_with(
            goto, task.I_WQC_FIRE, timeout=12
        )
        task.ui_get_current_page.assert_called_once_with()
        task.ui_goto.assert_called_once_with(page_exploration)


if __name__ == '__main__':
    unittest.main()
