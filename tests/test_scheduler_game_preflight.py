import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from script import Script


class SchedulerGamePreflightTest(unittest.TestCase):
    @staticmethod
    def make_script(game_is_running: bool, first_task: bool = True):
        script = Script.__new__(Script)
        script.config_name = 'test'
        script._emulator_down = False
        script._queue_owner_needs_restart = False
        script.is_first_task = first_task
        script.__dict__['device'] = Mock()
        script.device.app_is_running.return_value = game_is_running
        script.__dict__['config'] = Mock()
        return script

    def test_regular_task_queues_restart_when_game_is_closed(self):
        script = self.make_script(game_is_running=False)
        config = script.config

        self.assertFalse(script._prepare_scheduled_task('ExperienceYoukai'))

        config.task_call.assert_called_once_with('Restart')

    def test_regular_task_starts_when_game_is_running(self):
        script = self.make_script(game_is_running=True)

        self.assertTrue(script._prepare_scheduled_task('GoldYoukai'))

        script.config.task_call.assert_not_called()

    def test_initial_restart_is_skipped_only_when_game_is_running(self):
        running = self.make_script(game_is_running=True)
        stopped = self.make_script(game_is_running=False)
        running_config = running.config
        stopped_config = stopped.config

        self.assertFalse(running._prepare_scheduled_task('Restart'))
        running_config.task_delay.assert_called_once_with(
            task='Restart', success=True, server=True)
        self.assertTrue(stopped._prepare_scheduled_task('Restart'))
        stopped_config.task_delay.assert_not_called()

    def test_new_queue_owner_restarts_emulator_and_logs_in_before_task(self):
        script = self.make_script(game_is_running=True)
        script._queue_owner_needs_restart = True
        script.instance_guard = SimpleNamespace(resource_key='MuMuPlayer-15.0-1')
        script.device.emulator_start.return_value = True
        script.run = Mock(return_value=True)

        self.assertTrue(script._prepare_scheduled_task('AreaBoss'))

        script.device.emulator_start.assert_called_once_with()
        script.run.assert_called_once_with('Restart')
        self.assertFalse(script._queue_owner_needs_restart)

    @patch('script.time.sleep', return_value=None)
    def test_queue_waiter_never_stops_shared_emulator(self, _sleep):
        script = self.make_script(game_is_running=True)
        script.instance_guard = Mock()
        script.instance_guard.try_acquire.side_effect = [False, True]
        script.instance_guard.last_acquire_was_new = True
        script.config.script.optimization.queue_mode = True
        script.config.start_watching = Mock()
        script.config.should_reload.return_value = False

        self.assertTrue(script._try_acquire_queue_token())

        script.device.emulator_stop.assert_not_called()
        self.assertTrue(script._queue_owner_needs_restart)


if __name__ == '__main__':
    unittest.main()
