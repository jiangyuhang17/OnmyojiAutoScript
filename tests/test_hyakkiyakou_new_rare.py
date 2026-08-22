import unittest

import cv2
import numpy as np

from oashya.labels import CLASSINDEX as CI

from tasks.Hyakkiyakou.agent.new_rare import NewRareRecognizer


SUPPORTED_NAMES = {
    "arakuro",
    "chenhui_ebisu",
    "fengyangjun",
    "fuso_kozen",
    "himiko",
    "ichikami",
    "kannazuki",
    "karuta",
    "kuzuha",
    "longjue",
    "mengshan_shiro",
    "mengyin_kocho",
    "shenniang_hoshiguma_dojo",
    "shikinshin",
    "shiyao_takiyashahime",
    "shiyue_kyuketsuhime",
    "taira_no_masakado",
    "tianhuo_suzuhikohime",
    "ur_youtouhime",
    "xinyou_inugami",
    "yaoyin_kinnara",
    "yaonian_enenra",
    "yuki_gozen",
    "yunjian_fugengaku",
}


class NewRareRecognizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = NewRareRecognizer()

    def render_template(self, name, x=400, y=200):
        template = next(item for item in self.recognizer.templates if item.name == name)
        asset = next(
            item
            for item in self.recognizer.asset_dir.glob(f"{name}_*.png")
            if cv2.imread(str(item), cv2.IMREAD_UNCHANGED).shape[:2]
            == (template.height, template.width)
        )
        rgba = cv2.imread(str(asset), cv2.IMREAD_UNCHANGED)
        image = np.full((720, 1280, 3), 127, dtype=np.uint8)
        crop = image[y:y + rgba.shape[0], x:x + rgba.shape[1]]
        crop[:] = rgba[:, :, :3]
        return image

    def test_supported_names_are_complete(self):
        self.assertEqual(set(self.recognizer.names), SUPPORTED_NAMES)

    def test_each_curated_shikigami_uses_rare_compatibility_class(self):
        for name in self.recognizer.names:
            with self.subTest(name=name):
                tracks = self.recognizer(self.render_template(name))
                expected_id = self.recognizer.TRACK_ID_BASE + self.recognizer.names.index(name)
                matching = [track for track in tracks if track[0] == expected_id]
                self.assertEqual(len(matching), 1)
                self.assertEqual(matching[0][1], CI.MIN_SP)
                self.assertEqual(len(tracks), 1)

    def test_plain_image_has_no_detection(self):
        image = np.full((720, 1280, 3), 127, dtype=np.uint8)

        self.assertEqual(self.recognizer(image), [])

    def test_non_normalized_image_is_ignored(self):
        image = np.full((360, 640, 3), 127, dtype=np.uint8)

        self.assertEqual(self.recognizer(image), [])


if __name__ == "__main__":
    unittest.main()
