import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tasks.KekkaiUtilize.script_task import ScriptTask


class KekkaiUtilizeTest(unittest.TestCase):
    def test_empty_friend_list_is_treated_as_opened(self):
        utilize_add = object()
        enter_realm = object()
        friend_group = object()
        zones_group = object()
        task = SimpleNamespace(
            I_UTILIZE_ADD=utilize_add,
            I_U_ENTER_REALM=enter_realm,
            I_UTILIZE_FRIEND_GROUP=friend_group,
            I_UTILIZE_ZONES_GROUP=zones_group,
            screenshot=Mock(),
            appear=Mock(side_effect=lambda image: image in (utilize_add, friend_group)),
            appear_then_click=Mock(),
        )

        result = ScriptTask.grown_goto_utilize(task)

        self.assertTrue(result)
        task.appear_then_click.assert_not_called()


if __name__ == '__main__':
    unittest.main()
