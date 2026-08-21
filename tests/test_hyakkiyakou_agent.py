import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from oashya.labels import CLASSINDEX as CI

from tasks.Hyakkiyakou.agent.agent import Agent
from tasks.Hyakkiyakou.script_task import ScriptTask
from tasks.Hyakkiyakou.slave.hya_slave import HyaBuff


def track(track_id, class_id, x=640):
    return (track_id, class_id, 0.9, x, 360, 100, 160, 0.0)


class HyakkiyakouAgentTest(unittest.TestCase):
    def setUp(self):
        self.empty_state = [250, 36, 10] + [HyaBuff.BUFF_STATE0] * 4
        self.prob_up_state = [250, 36, 10, HyaBuff.BUFF_STATE6] + [HyaBuff.BUFF_STATE0] * 3
        self.ssr = CI.MIN_SSR
        self.sp = CI.MIN_SP

    def test_targets_rare_without_active_probability_buff(self):
        tracks = [track(1, CI.MIN_SR, 1100), track(2, self.ssr, 700)]

        self.assertEqual(Agent.select_target(tracks, self.empty_state)[0], 2)

    def test_ignores_non_rare_shikigami_and_unwanted_buffs(self):
        tracks = [track(1, CI.MIN_SR), track(2, CI.BUFF_002), track(3, CI.BUFF_005)]

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

    def test_prob_up_waits_until_early_threshold_without_rare_target(self):
        right_tracks = [track(1, CI.BUFF_006, 721)]
        left_tracks = [track(1, CI.BUFF_006, 720)]

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

        selected = Agent.select_target(tracks, self.empty_state)

        self.assertIsNone(selected)

    def test_freeze_is_always_ignored(self):
        tracks = [track(1, CI.BUFF_005), track(2, self.ssr)]

        selected = Agent.select_target(tracks, self.prob_up_state)

        self.assertEqual(selected[1], self.ssr)

    def test_prefers_rightmost_rare_target(self):
        tracks = [track(1, self.ssr, 500), track(2, self.sp, 1000)]

        selected = Agent.select_target(tracks, self.prob_up_state)

        self.assertEqual(selected[0], 2)

    def test_click_coordinates_stay_inside_screen(self):
        agent = Agent()
        tracks = [(1, self.ssr, 0.9, 5, 20, 100, 160, -1.0)]

        action = agent.decision(tracks, self.empty_state)

        self.assertEqual(action[:2], [0, 0])
        self.assertTrue(action[2])

    def test_rare_target_uses_two_click_burst(self):
        focus = SimpleNamespace(_class=self.ssr, _cx=900)

        self.assertEqual(ScriptTask._action_click_count(focus, self.empty_state), 2)

    def test_buff_target_uses_single_click(self):
        focus = SimpleNamespace(_class=CI.BUFF_006, _cx=900)

        self.assertEqual(ScriptTask._action_click_count(focus, self.empty_state), 1)

    def test_rare_target_on_left_uses_single_click(self):
        focus = SimpleNamespace(_class=self.sp, _cx=479)

        self.assertEqual(ScriptTask._action_click_count(focus, self.empty_state), 1)

    def test_rare_target_uses_single_click_when_beans_are_low(self):
        focus = SimpleNamespace(_class=self.sp, _cx=900)
        low_bean_state = [15, 36, 10] + [HyaBuff.BUFF_STATE0] * 4

        self.assertEqual(ScriptTask._action_click_count(focus, low_bean_state), 1)

    def test_rare_burst_clicks_twice_and_reports_total_beans(self):
        action = [900, 360, True, 10]
        fast_click = Mock()
        task = SimpleNamespace(
            agent=SimpleNamespace(focus=SimpleNamespace(_class=self.sp, _cx=900)),
            _config=SimpleNamespace(
                debug_config=SimpleNamespace(hya_control_method='minitouch')
            ),
            fast_click=fast_click,
            _action_click_count=ScriptTask._action_click_count,
        )

        ScriptTask.do_action(task, action, self.empty_state)

        self.assertEqual(fast_click.call_count, 2)
        self.assertEqual(action[3], 20)


if __name__ == '__main__':
    unittest.main()
