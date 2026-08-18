import unittest
from unittest.mock import Mock

from tasks.Secret.script_task import ScriptTask


class SecretEntryTest(unittest.TestCase):
    def make_task(self, *, battle=False, prepare=False, placement=False):
        task = ScriptTask.__new__(ScriptTask)
        task.screenshot = Mock()
        task.is_in_battle = Mock(return_value=battle)
        task.is_in_prepare = Mock(return_value=prepare)
        task.appear = Mock(return_value=placement)
        return task

    def test_active_battle_is_taken_over(self):
        task = self.make_task(battle=True)

        self.assertEqual(task.wait_entry_state(timeout=0), 'battle')

    def test_prepare_page_is_taken_over(self):
        task = self.make_task(prepare=True)

        self.assertEqual(task.wait_entry_state(timeout=0), 'battle')

    def test_secret_zone_page_is_accepted(self):
        task = self.make_task(placement=True)

        self.assertEqual(task.wait_entry_state(timeout=0), 'zones')

    def test_unknown_page_times_out(self):
        task = self.make_task()

        self.assertEqual(task.wait_entry_state(timeout=0), 'unknown')


if __name__ == '__main__':
    unittest.main()
