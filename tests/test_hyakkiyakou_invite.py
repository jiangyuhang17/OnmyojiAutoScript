import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tasks.Hyakkiyakou.slave.hya_slave import HyaSlave


class HyakkiyakouInviteTest(unittest.TestCase):
    def test_invite_friend_stops_when_tab_switch_fails(self):
        task = SimpleNamespace(_switch_friend_tab=Mock(return_value=False))

        result = HyaSlave._invite_friend(task, object(), object())

        self.assertFalse(result)

    def test_no_available_friend_closes_dialog_and_continues(self):
        same_1, remote_1, guild_1 = object(), object(), object()
        same_2, remote_2, guild_2 = object(), object(), object()
        task = SimpleNamespace(
            I_HINVITE=object(),
            I_CHECK_INVITATION=object(),
            I_ENSURE_RECALL=object(),
            I_FRIEND_SAME_1=same_1,
            I_FRIEND_REMOTE_1=remote_1,
            I_FRIEND_RYOU_1=guild_1,
            I_FRIEND_SAME_2=same_2,
            I_FRIEND_REMOTE_2=remote_2,
            I_FRIEND_RYOU_2=guild_2,
            ui_click=Mock(),
            appear=Mock(return_value=False),
            _invite_friend=Mock(return_value=False),
            _close_invitation=Mock(return_value=True),
        )

        result = HyaSlave.invite_friend(task)

        self.assertFalse(result)
        self.assertEqual(task._invite_friend.call_count, 3)
        task._close_invitation.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
