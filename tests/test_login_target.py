import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tasks.Component.SwitchAccount.login_account import LoginAccount
from tasks.Restart.login import LoginHandler


class LoginTargetTest(unittest.TestCase):
    @staticmethod
    def make_handler(server='永生之海', account='微信', android=True, enabled=True):
        handler = LoginHandler.__new__(LoginHandler)
        handler.config = SimpleNamespace(
            restart=SimpleNamespace(
                login_character_config=SimpleNamespace(
                    character='',
                    auto_switch_enable=enabled,
                    account=account,
                    server=server,
                    apple_or_android=android,
                ),
            ),
        )
        handler.device = Mock()
        return handler

    def test_server_name_matching_tolerates_one_ocr_error(self):
        self.assertTrue(LoginAccount.server_name_matches('永生之海', '永生之海'))
        self.assertTrue(LoginAccount.server_name_matches('永生之侮', '永生之海'))
        self.assertTrue(LoginAccount.server_name_matches('结伴同游', '网易-结伴同游'))
        self.assertFalse(LoginAccount.server_name_matches('森之王', '永生之海'))

    def test_matching_server_does_not_open_account_switcher(self):
        handler = self.make_handler()
        helper = Mock()
        helper.get_svr_name.return_value = '永生之海'
        helper.server_name_matches.side_effect = LoginAccount.server_name_matches
        handler._login_account_helper = Mock(return_value=helper)

        self.assertTrue(handler._ensure_login_target())

        helper.login.assert_not_called()

    def test_mismatched_server_switches_account_platform_and_server(self):
        handler = self.make_handler(server='铃心踏焰', account='网易邮箱', android=False)
        helper = Mock()
        helper.get_svr_name.side_effect = ['高天原', '铃心踏焰']
        helper.server_name_matches.side_effect = LoginAccount.server_name_matches
        helper.login.return_value = True
        handler._login_account_helper = Mock(return_value=helper)

        self.assertTrue(handler._ensure_login_target())

        account_info = helper.login.call_args.args[0]
        self.assertEqual(account_info.account, '网易邮箱')
        self.assertEqual(account_info.svr, '铃心踏焰')
        self.assertFalse(account_info.apple_or_android)

    def test_failed_server_verification_stops_login(self):
        handler = self.make_handler(server='森之王')
        helper = Mock()
        helper.get_svr_name.side_effect = ['永生之海', '永生之海']
        helper.server_name_matches.side_effect = LoginAccount.server_name_matches
        helper.login.return_value = True
        handler._login_account_helper = Mock(return_value=helper)

        self.assertFalse(handler._ensure_login_target())


if __name__ == '__main__':
    unittest.main()
