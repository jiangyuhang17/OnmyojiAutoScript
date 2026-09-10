import tempfile
import unittest
from pathlib import Path

from module.config.instance_guard import InstanceGuard


class InstanceGuardGroupTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name)
        self.guards = []

    def tearDown(self):
        for guard in self.guards:
            guard.remove_from_queue()
        self.temp_dir.cleanup()

    def guard(self, name, emulator):
        guard = InstanceGuard(name, resource_key=emulator, state_dir=self.state_dir)
        self.guards.append(guard)
        return guard

    def test_same_emulator_is_mutually_exclusive(self):
        and1 = self.guard('and1', 'MuMuPlayer-15.0-1')
        and2 = self.guard('and2', 'MuMuPlayer-15.0-1')

        self.assertTrue(and1.try_acquire())
        self.assertTrue(and1.last_acquire_was_new)
        self.assertFalse(and2.try_acquire())

    def test_different_emulators_can_run_in_parallel(self):
        android = self.guard('and1', 'MuMuPlayer-15.0-1')
        ios = self.guard('ios1', 'MuMuPlayer-15.0-0')

        self.assertTrue(android.try_acquire())
        self.assertTrue(ios.try_acquire())

    def test_release_hands_token_to_next_waiter(self):
        and1 = self.guard('and1', 'MuMuPlayer-15.0-1')
        and2 = self.guard('and2', 'MuMuPlayer-15.0-1')

        self.assertTrue(and1.try_acquire())
        self.assertFalse(and2.try_acquire())
        and1.release()

        self.assertTrue(and2.try_acquire())
        self.assertTrue(and2.last_acquire_was_new)


if __name__ == '__main__':
    unittest.main()
