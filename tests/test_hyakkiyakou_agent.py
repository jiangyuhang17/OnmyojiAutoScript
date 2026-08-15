import unittest

from oashya.labels import CLASSINDEX as CI

from tasks.Hyakkiyakou.agent.agent import Agent
from tasks.Hyakkiyakou.slave.hya_slave import HyaBuff


def track(track_id, class_id, x=640):
    return (track_id, class_id, 0.9, x, 360, 100, 160, 0.0)


class HyakkiyakouAgentTest(unittest.TestCase):
    def setUp(self):
        self.empty_state = [250, 36, 10] + [HyaBuff.BUFF_STATE0] * 4
        self.prob_up_state = [250, 36, 10, HyaBuff.BUFF_STATE6] + [HyaBuff.BUFF_STATE0] * 3
        self.ssr = CI.MIN_SSR
        self.sp = CI.MIN_SP

    def test_only_targets_ssr_or_sp_shikigami(self):
        tracks = [track(1, CI.MIN_SR, 1100), track(2, self.ssr, 700)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[0], 2)

    def test_ignores_non_rare_shikigami_and_unwanted_buffs(self):
        tracks = [track(1, CI.MIN_SR), track(2, CI.BUFF_002), track(3, CI.BUFF_007)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertIsNone(selected)

    def test_normal_buff_priority_precedes_rare_target(self):
        tracks = [
            track(1, self.sp),
            track(2, CI.BUFF_004),
            track(3, CI.BUFF_003),
            track(4, CI.BUFF_006),
        ]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[1], CI.BUFF_006)

    def test_prob_up_is_hit_immediately_when_rare_target_is_visible(self):
        tracks = [track(1, CI.BUFF_006, 1100), track(2, self.ssr, 900)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[1], CI.BUFF_006)

    def test_prob_up_waits_until_left_half_without_rare_target(self):
        right_tracks = [track(1, CI.BUFF_006, 900)]
        left_tracks = [track(1, CI.BUFF_006, 640)]

        self.assertIsNone(Agent.select_target(right_tracks, self.empty_state))
        self.assertEqual(Agent.select_target(left_tracks, self.empty_state)[1], CI.BUFF_006)

    def test_other_buffs_can_be_hit_while_prob_up_waits(self):
        tracks = [track(1, CI.BUFF_006, 900), track(2, CI.BUFF_003, 800)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[1], CI.BUFF_003)

    def test_speed_precedes_extra_beans(self):
        tracks = [track(1, CI.BUFF_004), track(2, CI.BUFF_003), track(3, self.ssr)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[1], CI.BUFF_003)

    def test_slow_has_top_priority_with_prob_up_and_rare_target(self):
        tracks = [track(1, CI.BUFF_006), track(2, CI.BUFF_002), track(3, self.ssr)]

        selected = Agent.select_target(tracks, self.prob_up_state)

        self.assertEqual(selected[1], CI.BUFF_002)

    def test_slow_is_ignored_without_active_prob_up(self):
        tracks = [track(1, CI.BUFF_002), track(2, self.ssr)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[1], self.ssr)

    def test_slow_is_ignored_without_rare_target(self):
        tracks = [track(1, CI.BUFF_002), track(2, CI.MIN_SR)]

        selected = Agent.select_target(tracks, self.prob_up_state)

        self.assertIsNone(selected)

    def test_freeze_is_always_ignored(self):
        tracks = [track(1, CI.BUFF_005), track(2, self.ssr)]

        selected = Agent.select_target(tracks, self.prob_up_state)

        self.assertEqual(selected[1], self.ssr)

    def test_prefers_rightmost_rare_target(self):
        tracks = [track(1, self.ssr, 500), track(2, self.sp, 1000)]

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertEqual(selected[0], 2)

    def test_click_coordinates_stay_inside_screen(self):
        agent = Agent()
        tracks = [(1, self.ssr, 0.9, 5, 20, 100, 160, -1.0)]

        action = agent.decision(tracks, self.empty_state)

        self.assertEqual(action[:2], [0, 0])
        self.assertTrue(action[2])


if __name__ == '__main__':
    unittest.main()
