import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from module.device.platform2.emulator_windows import EmulatorInstance
from module.device.platform2.platform_windows import PlatformWindows


class EmulatorPlatformTest(unittest.TestCase):
    @staticmethod
    def mumu_instance(version=15, instance_id=0):
        return EmulatorInstance(
            serial=f'127.0.0.1:{16384 + 32 * instance_id}',
            name=f'MuMuPlayer-{version}.0-{instance_id}',
            path='C:/MuMuPlayer/nx_main/MuMuNxMain.exe',
        )

    def test_mumu_instance_name_exposes_engine_version(self):
        instance = self.mumu_instance(version=15, instance_id=3)

        self.assertEqual(instance.MuMuPlayer12_id, 3)
        self.assertEqual(instance.MuMuPlayer_version, 15)

    @patch('module.device.platform2.platform_windows.os.path.exists', return_value=True)
    def test_mumu_nx_launches_device_process_directly(self, _exists):
        platform = PlatformWindows.__new__(PlatformWindows)
        platform.config = SimpleNamespace(
            script=SimpleNamespace(
                device=SimpleNamespace(
                    emulator_window_minimize=False,
                    run_background_only=False,
                ),
            ),
        )
        platform.execute = Mock(return_value=SimpleNamespace(pid=1234))

        platform._emulator_start(self.mumu_instance())

        platform.execute.assert_called_once_with(
            '"C:/MuMuPlayer/nx_device/15.0/shell/MuMuNxDevice.exe" '
            '-v 0 --vm MuMuPlayer-15.0-0',
            show_window=True,
        )
        self.assertEqual(platform._emulator_start_process_pid, 1234)

    @patch('module.device.platform2.platform_windows.os.path.exists', return_value=True)
    def test_mumu_nx_direct_launch_includes_nonzero_instance_id(self, _exists):
        platform = PlatformWindows.__new__(PlatformWindows)
        platform.config = SimpleNamespace(
            script=SimpleNamespace(
                device=SimpleNamespace(
                    emulator_window_minimize=False,
                    run_background_only=False,
                ),
            ),
        )
        platform.execute = Mock(return_value=SimpleNamespace(pid=5678))

        platform._emulator_start(self.mumu_instance(instance_id=1))

        platform.execute.assert_called_once_with(
            '"C:/MuMuPlayer/nx_device/15.0/shell/MuMuNxDevice.exe" '
            '-v 1 --vm MuMuPlayer-15.0-1',
            show_window=True,
        )
        self.assertEqual(platform._emulator_start_process_pid, 5678)

    @patch.object(PlatformWindows, '_mumu_nx_force_stop')
    @patch.object(PlatformWindows, '_mumu_nx_info')
    def test_mumu_nx_stop_skips_graceful_shutdown(self, info, force_stop):
        instance = self.mumu_instance()
        instance_info = {'index': '0', 'android_version': '15.0', 'pid': 6640}
        info.return_value = instance_info
        platform = PlatformWindows.__new__(PlatformWindows)
        platform.execute = Mock()

        platform._emulator_stop(instance)

        platform.execute.assert_not_called()
        force_stop.assert_called_once_with(instance, instance_info)

    @patch('module.device.platform2.platform_windows.psutil.Process')
    def test_mumu_nx_force_stop_kills_main_process_directly(self, process_factory):
        process = process_factory.return_value
        process.pid = 6640
        process.name.return_value = 'MuMuNxDevice.exe'
        process.cmdline.return_value = [
            'C:/MuMuPlayer/nx_device/15.0/shell/MuMuNxDevice.exe',
            '--vm',
            'MuMuPlayer-15.0-0',
        ]
        info = {
            'index': '0',
            'android_version': '15.0',
            'pid': 6640,
            'is_process_started': True,
        }

        PlatformWindows._mumu_nx_force_stop(self.mumu_instance(), info)

        process_factory.assert_called_once_with(6640)
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10)

    @patch('module.device.platform2.platform_windows.psutil.process_iter')
    @patch('module.device.platform2.platform_windows.psutil.Process')
    def test_mumu_nx_force_stop_recovers_from_stale_manager_pid(
            self, process_factory, process_iter):
        stale = process_factory.return_value
        stale.name.return_value = 'unrelated.exe'
        stale.cmdline.return_value = ['unrelated.exe']
        actual = Mock()
        actual.pid = 13832
        actual.name.return_value = 'MuMuNxDevice.exe'
        actual.cmdline.return_value = [
            'C:/MuMuPlayer/nx_device/15.0/shell/MuMuNxDevice.exe',
            '--vm',
            'MuMuPlayer-15.0-0',
        ]
        process_iter.return_value = [actual]
        info = {'index': '0', 'android_version': '15.0', 'pid': 14200}

        PlatformWindows._mumu_nx_force_stop(self.mumu_instance(), info)

        actual.kill.assert_called_once_with()
        actual.wait.assert_called_once_with(timeout=10)


if __name__ == '__main__':
    unittest.main()
