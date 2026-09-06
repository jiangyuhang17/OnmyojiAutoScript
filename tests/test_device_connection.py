import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from module.device.connection import Connection
from module.exception import EmulatorNotRunningError
from module.map.map_grids import SelectedGrids


class DeviceConnectionTest(unittest.TestCase):
    @staticmethod
    def connection(serial, devices):
        connection = Connection.__new__(Connection)
        connection.serial = serial
        connection.config = SimpleNamespace(
            script=SimpleNamespace(
                device=SimpleNamespace(serial=serial),
            ),
        )
        connection.list_device = Mock(return_value=SelectedGrids(devices))
        return connection

    def test_detect_device_restarts_configured_offline_emulator(self):
        device = SimpleNamespace(serial='127.0.0.1:16384', status='offline')
        connection = self.connection(device.serial, [device])

        with self.assertRaises(EmulatorNotRunningError):
            connection.detect_device()

    def test_detect_device_ignores_other_offline_devices(self):
        current = SimpleNamespace(serial='127.0.0.1:16384', status='device')
        other = SimpleNamespace(serial='127.0.0.1:16416', status='offline')
        connection = self.connection(current.serial, [current, other])

        connection.detect_device()

    def test_first_adb_read_timeout_restarts_local_emulator(self):
        connection = Connection.__new__(Connection)
        connection.serial = '127.0.0.1:16384'
        connection.adb_reconnect = Mock()
        connection.emulator_start = Mock(return_value=True)

        connection.adb_recover(Exception('adb read timeout'))

        connection.adb_reconnect.assert_not_called()
        connection.emulator_start.assert_called_once_with()

    def test_emulator_recover_rejects_network_device(self):
        connection = Connection.__new__(Connection)
        connection.serial = '192.168.1.10:5555'
        connection.emulator_start = Mock(return_value=True)

        self.assertFalse(connection.emulator_recover('test failure'))

        connection.emulator_start.assert_not_called()

    def test_failed_local_emulator_restart_falls_back_to_adb_reconnect(self):
        connection = Connection.__new__(Connection)
        connection.serial = '127.0.0.1:16384'
        connection.adb_reconnect = Mock()
        connection.emulator_start = Mock(return_value=False)

        connection.adb_recover(Exception('adb read timeout'))

        connection.adb_reconnect.assert_called_once_with()
        connection.emulator_start.assert_called_once_with()

    def test_network_device_adb_read_timeout_only_reconnects_adb(self):
        connection = Connection.__new__(Connection)
        connection.serial = '192.168.1.10:5555'
        connection.adb_reconnect = Mock()
        connection.emulator_start = Mock(return_value=True)

        connection.adb_recover(Exception('adb read timeout'))

        connection.adb_reconnect.assert_called_once_with()
        connection.emulator_start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
