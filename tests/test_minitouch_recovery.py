import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from module.device.method.minitouch import (
    MinitouchNotInstalledError,
    retry,
)


class MinitouchRecoveryTest(unittest.TestCase):
    def test_second_empty_response_restarts_emulator(self):
        attempts = 0

        @retry
        def operation(_device):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise MinitouchNotInstalledError('empty')
            return 'recovered'

        device = SimpleNamespace(
            _minitouch_port=0,
            install_uiautomator2=Mock(),
            emulator_recover=Mock(return_value=True),
        )
        with patch('module.device.method.minitouch.retry_sleep'), \
                patch('module.device.method.minitouch.del_cached_property'):
            result = operation(device)

        self.assertEqual(result, 'recovered')
        self.assertEqual(attempts, 3)
        device.install_uiautomator2.assert_called_once_with()
        device.emulator_recover.assert_called_once_with(
            'Minitouch returned empty data 2 times'
        )


if __name__ == '__main__':
    unittest.main()
