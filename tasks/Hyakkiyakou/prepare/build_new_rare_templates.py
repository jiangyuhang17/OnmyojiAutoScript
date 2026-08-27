"""Build masked templates for new rare shikigami from real Hyakki captures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CAPTURE_DIR = ROOT / "log" / "hya" / "20260815T23"
OUTPUT_DIR = ROOT / "tasks" / "Hyakkiyakou" / "new_rare_templates"
PUBLIC_ROOT = (
    ROOT
    / "tasks"
    / "Hyakkiyakou"
    / "dataset"
    / "new_shikigami"
    / "public_hyakki"
    / "BV1uwgP69Eje_rewards_v3"
)
PUBLIC_V4_ROOT = PUBLIC_ROOT.with_name("BV1uwgP69Eje_rewards_v4")


@dataclass(frozen=True)
class Annotation:
    source: str
    filename: str
    bbox: tuple[int, int, int, int]
    include: tuple[tuple[int, int, int, int], ...] = ()


SOURCES = {
    "local": {
        "directory": CAPTURE_DIR,
        "background": None,
    },
    "local_20260816T17": {
        "directory": ROOT / "log" / "hya" / "20260816T17",
        "background": None,
    },
    "local_20260816T22": {
        "directory": ROOT / "log" / "hya" / "20260816T22",
        "background": None,
    },
    "local_20260819T03": {
        "directory": ROOT / "log" / "hya" / "20260819T03",
        "background": None,
    },
    "local_20260820T04": {
        "directory": ROOT / "log" / "hya" / "20260820T04",
        "background": None,
    },
    "local_20260820T23": {
        "directory": ROOT / "log" / "hya" / "20260820T23",
        "background": None,
    },
    "local_20260821T03": {
        "directory": ROOT / "log" / "hya" / "20260821T03",
        "background": None,
    },
    "local_20260821T04": {
        "directory": ROOT / "log" / "hya" / "20260821T04",
        "background": None,
    },
    "local_20260821T05": {
        "directory": ROOT / "log" / "hya" / "20260821T05",
        "background": None,
    },
    "local_20260821T16": {
        "directory": ROOT / "log" / "hya" / "20260821T16",
        "background": None,
    },
    "local_20260822T03": {
        "directory": ROOT / "log" / "hya" / "20260822T03",
        "background": None,
    },
    "local_20260826T02": {
        "directory": ROOT / "log" / "hya" / "20260826T02",
        "background": None,
    },
    "local_20260826T05": {
        "directory": ROOT / "log" / "hya" / "20260826T05",
        "background": None,
    },
    "local_20260827T14": {
        "directory": ROOT / "log" / "hya" / "20260827T14",
        "background": None,
    },
    "public_20260814": {
        "directory": PUBLIC_ROOT / "targets" / "ssr" / "taira_no_masakado",
        "background": PUBLIC_ROOT / "events" / "event_0277_07751.744.jpg",
    },
    "public_kannazuki": {
        "directory": (
            ROOT
            / "tasks"
            / "Hyakkiyakou"
            / "dataset"
            / "new_shikigami"
            / "public_hyakki"
            / "BV1w9ZvBjEGi_rewards_v3"
            / "targets"
            / "ssr"
            / "kannazuki"
        ),
        "background": (
            ROOT
            / "tasks"
            / "Hyakkiyakou"
            / "dataset"
            / "new_shikigami"
            / "public_hyakki"
            / "BV1w9ZvBjEGi_rewards_v3"
            / "targets"
            / "ssr"
            / "kannazuki"
            / "event_0004_00_00038.400.jpg"
        ),
    },
    "public_longjue": {
        "directory": PUBLIC_V4_ROOT / "longjue_pre",
        "background": None,
    },
    "public_yaoyin": {
        "directory": PUBLIC_V4_ROOT / "yaoyin_pre",
        "background": None,
    },
    "public_mengyin": {
        "directory": PUBLIC_V4_ROOT / "mengyin_1",
        "background": None,
    },
    "public_shiyue": {
        "directory": PUBLIC_V4_ROOT / "shiyue_pre",
        "background": None,
    },
    "public_yaonian": {
        "directory": PUBLIC_ROOT / "targets" / "sp" / "yaonian_enenra",
        "background": None,
    },
}

# Coordinates are target-only regions in normalized 1280x720 captures.
ANNOTATIONS = {
    "ur_youtouhime": [
        Annotation(
            "local_20260822T03",
            "hya_1787342299724.jpg",
            (1020, 330, 1280, 610),
            ((1050, 360, 1280, 590),),
        ),
        Annotation(
            "local_20260822T03",
            "hya_1787342302129.jpg",
            (590, 330, 825, 610),
            ((610, 350, 815, 590),),
        ),
        Annotation(
            "local_20260822T03",
            "hya_1787342303374.jpg",
            (285, 330, 525, 610),
            (
                (300, 350, 410, 590),
                (390, 350, 515, 425),
            ),
        ),
        Annotation(
            "local_20260826T02",
            "hya_1787682391373.jpg",
            (995, 345, 1280, 610),
            ((1015, 365, 1275, 595),),
        ),
    ],
    "longyin_suzuka_gozen": [
        Annotation(
            "local_20260826T05",
            "hya_1787692144597.jpg",
            (970, 255, 1280, 610),
            ((990, 275, 1280, 600),),
        ),
        Annotation(
            "local_20260826T05",
            "hya_1787692145257.jpg",
            (820, 255, 1080, 600),
            ((840, 275, 1065, 590),),
        ),
        Annotation(
            "local_20260826T05",
            "hya_1787692146002.jpg",
            (725, 250, 1040, 600),
            ((750, 265, 960, 590),),
        ),
        Annotation(
            "local_20260826T05",
            "hya_1787692146640.jpg",
            (600, 260, 875, 610),
            ((620, 275, 840, 600),),
        ),
        Annotation(
            "local_20260826T05",
            "hya_1787692147311.jpg",
            (505, 270, 760, 610),
            ((525, 285, 720, 600),),
        ),
        Annotation(
            "local_20260826T05",
            "hya_1787692148751.jpg",
            (285, 260, 555, 610),
            ((305, 280, 535, 600),),
        ),
        Annotation(
            "local_20260826T05",
            "hya_1787692149447.jpg",
            (150, 260, 400, 610),
            ((170, 280, 380, 600),),
        ),
    ],
    "fuso_kozen": [
        Annotation("local_20260821T05", "hya_1787260314501.jpg", (1090, 280, 1280, 610)),
        Annotation("local_20260821T05", "hya_1787260316128.jpg", (900, 275, 1190, 610)),
        Annotation("local_20260821T05", "hya_1787260317563.jpg", (690, 275, 1000, 610)),
        Annotation("local_20260821T05", "hya_1787260323094.jpg", (125, 270, 465, 610)),
    ],
    "himiko": [
        Annotation("local_20260821T03", "hya_1787254437297.jpg", (1000, 245, 1280, 600)),
        Annotation("local_20260821T03", "hya_1787254438817.jpg", (630, 245, 1010, 600)),
        Annotation("local_20260821T03", "hya_1787254440417.jpg", (330, 245, 765, 600)),
        Annotation("local_20260821T03", "hya_1787254441997.jpg", (60, 245, 515, 600)),
    ],
    "mengshan_shiro": [
        Annotation("local_20260821T04", "hya_1787256929456.jpg", (980, 250, 1280, 610)),
        Annotation("local_20260821T04", "hya_1787256930942.jpg", (760, 245, 1060, 610)),
    ],
    "tianhuo_suzuhikohime": [
        Annotation("local_20260821T04", "hya_1787257278204.jpg", (775, 280, 1060, 600)),
        Annotation("local_20260821T04", "hya_1787257279436.jpg", (525, 275, 780, 600)),
        Annotation("local_20260821T04", "hya_1787257280751.jpg", (250, 270, 540, 600)),
    ],
    "yuki_gozen": [
        Annotation(
            "local_20260821T16",
            "hya_1787301340787.jpg",
            (950, 280, 1240, 540),
            ((975, 300, 1230, 475),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301341992.jpg",
            (780, 280, 1070, 540),
            ((805, 300, 1055, 475),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301343153.jpg",
            (580, 280, 860, 540),
            ((610, 300, 840, 475),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301344393.jpg",
            (370, 280, 650, 540),
            ((400, 300, 630, 475),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301345567.jpg",
            (120, 280, 430, 540),
            ((145, 300, 410, 475),),
        ),
    ],
    "shikinshin": [
        Annotation(
            "local_20260820T23",
            "hya_1787239018415.jpg",
            (895, 295, 1160, 575),
            ((910, 315, 1145, 540),),
        ),
        Annotation(
            "local_20260820T23",
            "hya_1787239019473.jpg",
            (685, 295, 980, 575),
            ((705, 315, 965, 540),),
        ),
        Annotation(
            "local_20260820T23",
            "hya_1787239020522.jpg",
            (565, 295, 810, 575),
            ((585, 315, 795, 540),),
        ),
        Annotation(
            "local_20260820T23",
            "hya_1787239021711.jpg",
            (390, 295, 715, 575),
            ((410, 315, 655, 540),),
        ),
    ],
    "kuzuha": [
        Annotation(
            "local_20260820T04",
            "hya_1787169966513.jpg",
            (480, 285, 765, 595),
            ((495, 305, 735, 575),),
        ),
        Annotation(
            "local_20260820T04",
            "hya_1787169967187.jpg",
            (305, 285, 675, 595),
            ((325, 305, 600, 575),),
        ),
        Annotation(
            "local_20260820T04",
            "hya_1787169967921.jpg",
            (185, 285, 565, 595),
            ((205, 305, 485, 575),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811176887.jpg",
            (1020, 280, 1280, 600),
            ((1040, 340, 1275, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811177743.jpg",
            (900, 280, 1240, 600),
            ((930, 340, 1220, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811178518.jpg",
            (750, 275, 1150, 600),
            ((760, 445, 1125, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811179379.jpg",
            (600, 275, 1000, 600),
            ((620, 430, 975, 585),),
        ),
    ],
    "ichikami": [
        Annotation(
            "local_20260819T03",
            "hya_1787080150148.jpg",
            (955, 325, 1215, 610),
            ((975, 335, 1190, 600),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787080150696.jpg",
            (885, 325, 1165, 610),
            ((900, 335, 1140, 600),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787080151685.jpg",
            (735, 325, 1000, 610),
            ((750, 335, 975, 600),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787080152205.jpg",
            (635, 325, 905, 610),
            ((650, 335, 880, 600),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787080153090.jpg",
            (500, 325, 735, 610),
            ((510, 335, 710, 600),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301878643.jpg",
            (820, 320, 1110, 590),
            ((850, 335, 1080, 535),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301880439.jpg",
            (530, 320, 800, 590),
            ((555, 340, 780, 520),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301881310.jpg",
            (370, 320, 650, 590),
            ((400, 340, 620, 535),),
        ),
    ],
    "karuta": [
        Annotation(
            "local_20260821T16",
            "hya_1787301979223.jpg",
            (800, 315, 970, 570),
            ((825, 335, 945, 550),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301980341.jpg",
            (620, 310, 790, 570),
            ((645, 330, 765, 550),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301981453.jpg",
            (430, 315, 600, 570),
            ((455, 335, 575, 550),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301982680.jpg",
            (250, 315, 420, 570),
            ((275, 335, 395, 550),),
        ),
        Annotation(
            "local_20260821T16",
            "hya_1787301983791.jpg",
            (55, 330, 225, 575),
            ((80, 350, 205, 555),),
        ),
    ],
    "arakuro": [
        Annotation("local", "hya_1786808142237.jpg", (745, 215, 990, 545)),
        Annotation("local", "hya_1786808143012.jpg", (585, 215, 890, 550)),
        Annotation("local", "hya_1786808143828.jpg", (425, 215, 745, 555)),
        Annotation(
            "local_20260827T14",
            "hya_1787811283734.jpg",
            (990, 285, 1280, 590),
            ((1000, 340, 1275, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811284225.jpg",
            (875, 285, 1160, 590),
            ((890, 340, 1145, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811284833.jpg",
            (760, 285, 1040, 590),
            ((775, 340, 1025, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811285413.jpg",
            (700, 285, 970, 590),
            ((720, 340, 950, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811286172.jpg",
            (590, 285, 815, 590),
            ((605, 340, 790, 585),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811287241.jpg",
            (475, 285, 650, 590),
            ((490, 340, 630, 515),),
        ),
        Annotation(
            "local_20260827T14",
            "hya_1787811288144.jpg",
            (275, 285, 455, 590),
            ((295, 340, 440, 550),),
        ),
    ],
    "shiyao_takiyashahime": [
        Annotation("local", "hya_1786808359590.jpg", (775, 245, 1105, 570)),
        Annotation("local", "hya_1786808360090.jpg", (690, 245, 1000, 570)),
        Annotation("local", "hya_1786808360677.jpg", (610, 240, 900, 570)),
        Annotation("local", "hya_1786808361788.jpg", (425, 245, 710, 570)),
    ],
    "chenhui_ebisu": [
        Annotation("local", "hya_1786808440713.jpg", (795, 245, 1125, 585)),
        Annotation("local", "hya_1786808441216.jpg", (715, 245, 1045, 585)),
        Annotation("local", "hya_1786808442193.jpg", (545, 245, 900, 585)),
    ],
    "yunjian_fugengaku": [
        Annotation(
            "local_20260816T22",
            "hya_1786890140614.jpg",
            (700, 325, 1010, 600),
            ((760, 330, 980, 535),),
        ),
        Annotation(
            "local_20260816T22",
            "hya_1786890141178.jpg",
            (570, 325, 900, 600),
            ((650, 330, 880, 535),),
        ),
        Annotation(
            "local_20260816T22",
            "hya_1786890152869.jpg",
            (695, 325, 975, 600),
            ((725, 330, 960, 535),),
        ),
    ],
    "taira_no_masakado": [
        Annotation("local_20260816T17", "hya_1786871038237.jpg", (535, 215, 865, 565)),
        Annotation("local_20260816T17", "hya_1786871039075.jpg", (385, 215, 720, 565)),
        Annotation("local_20260816T17", "hya_1786871039898.jpg", (245, 215, 600, 570)),
        Annotation("local_20260816T17", "hya_1786871040684.jpg", (115, 215, 475, 570)),
        Annotation(
            "public_20260814",
            "event_0277_00_07748.405.jpg",
            (500, 270, 810, 565),
            ((600, 315, 800, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_01_07748.882.jpg",
            (445, 275, 735, 565),
            ((590, 315, 730, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_02_07749.359.jpg",
            (360, 275, 660, 565),
            ((450, 315, 650, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_03_07749.836.jpg",
            (250, 280, 555, 565),
            ((390, 315, 550, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_04_07750.313.jpg",
            (80, 270, 500, 565),
            ((240, 315, 480, 455),),
        ),
        Annotation(
            "public_20260814",
            "event_0277_05_07750.790.jpg",
            (100, 275, 445, 565),
            ((240, 315, 440, 455),),
        ),
    ],
    "kannazuki": [
        Annotation("local_20260816T17", "hya_1786871141282.jpg", (905, 220, 1190, 595)),
        Annotation("local_20260816T17", "hya_1786871141966.jpg", (715, 220, 1010, 595)),
        Annotation("local_20260816T17", "hya_1786871142755.jpg", (590, 220, 890, 595)),
        Annotation("local_20260816T17", "hya_1786871143453.jpg", (450, 220, 755, 595)),
        Annotation("local_20260816T17", "hya_1786871144166.jpg", (370, 220, 680, 595)),
        Annotation(
            "public_kannazuki",
            "event_0004_04_00039.467.jpg",
            (900, 285, 1180, 590),
            ((930, 325, 1160, 565),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_06_00040.000.jpg",
            (880, 285, 1155, 590),
            ((910, 325, 1140, 565),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_08_00040.533.jpg",
            (780, 285, 1060, 590),
            ((805, 325, 1045, 565),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_10_00041.067.jpg",
            (690, 285, 965, 585),
            ((755, 340, 950, 555),),
        ),
        Annotation(
            "public_kannazuki",
            "event_0004_12_00041.600.jpg",
            (620, 290, 905, 590),
            ((650, 335, 890, 565),),
        ),
    ],
    "shenniang_hoshiguma_dojo": [
        Annotation("local_20260816T17", "hya_1786871239181.jpg", (290, 360, 465, 580)),
        Annotation("local_20260816T17", "hya_1786871239826.jpg", (175, 355, 350, 580)),
        Annotation("local_20260816T17", "hya_1786871240563.jpg", (45, 350, 220, 580)),
    ],
    "fengyangjun": [
        Annotation("local_20260816T17", "hya_1786871248295.jpg", (930, 330, 1280, 590)),
        Annotation("local_20260816T17", "hya_1786871248958.jpg", (825, 330, 1195, 590)),
        Annotation("local_20260816T17", "hya_1786871249620.jpg", (700, 330, 1085, 590)),
        Annotation(
            "local_20260822T03",
            "hya_1787339489340.jpg",
            (800, 320, 1190, 600),
            ((800, 320, 1148, 600),),
        ),
        Annotation(
            "local_20260822T03",
            "hya_1787339490349.jpg",
            (625, 320, 1015, 600),
            ((625, 320, 960, 600),),
        ),
        Annotation(
            "local_20260822T03",
            "hya_1787339491213.jpg",
            (445, 320, 825, 600),
            ((475, 320, 780, 600),),
        ),
        Annotation(
            "local_20260822T03",
            "hya_1787339492146.jpg",
            (285, 320, 675, 600),
            ((285, 320, 625, 600),),
        ),
    ],
    "xinyou_inugami": [
        Annotation("local_20260816T17", "hya_1786871176114.jpg", (425, 365, 615, 575)),
        Annotation("local_20260816T17", "hya_1786871177388.jpg", (220, 365, 450, 580)),
        Annotation("local_20260816T17", "hya_1786871178077.jpg", (115, 365, 345, 580)),
        Annotation(
            "local_20260819T03",
            "hya_1787081294876.jpg",
            (1045, 345, 1280, 605),
            ((1060, 355, 1275, 595),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787081295446.jpg",
            (960, 345, 1220, 605),
            ((975, 355, 1205, 595),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787081296262.jpg",
            (795, 345, 1035, 605),
            ((810, 355, 1020, 595),),
        ),
        Annotation(
            "local_20260819T03",
            "hya_1787081297762.jpg",
            (530, 345, 765, 605),
            ((545, 355, 750, 595),),
        ),
    ],
    "longjue": [
        Annotation(
            "public_longjue",
            "frame_05621.500.jpg",
            (760, 245, 1080, 585),
            ((790, 270, 1060, 555),),
        ),
        Annotation(
            "public_longjue",
            "frame_05622.000.jpg",
            (670, 245, 1000, 585),
            ((710, 270, 965, 555),),
        ),
        Annotation(
            "public_longjue",
            "frame_05622.500.jpg",
            (575, 245, 905, 590),
            ((610, 270, 870, 560),),
        ),
        Annotation(
            "public_longjue",
            "frame_05623.000.jpg",
            (485, 245, 820, 595),
            ((520, 270, 785, 565),),
        ),
    ],
    "yaoyin_kinnara": [
        Annotation(
            "public_yaoyin",
            "frame_06058.500.jpg",
            (810, 245, 1140, 590),
            ((875, 270, 1115, 555),),
        ),
        Annotation(
            "public_yaoyin",
            "frame_06059.000.jpg",
            (700, 245, 1030, 590),
            ((760, 270, 1000, 555),),
        ),
        Annotation(
            "public_yaoyin",
            "frame_06061.000.jpg",
            (360, 245, 690, 590),
            ((405, 270, 630, 550),),
        ),
        Annotation(
            "public_yaoyin",
            "frame_06062.000.jpg",
            (175, 245, 500, 590),
            ((210, 270, 440, 550),),
        ),
    ],
    "mengyin_kocho": [
        Annotation(
            "public_mengyin",
            "frame_03458.000.jpg",
            (900, 250, 1230, 575),
            ((940, 280, 1200, 550),),
        ),
        Annotation(
            "public_mengyin",
            "frame_03458.500.jpg",
            (805, 250, 1135, 575),
            ((850, 280, 1105, 550),),
        ),
        Annotation(
            "public_mengyin",
            "frame_03459.000.jpg",
            (735, 250, 1060, 575),
            ((780, 280, 1030, 550),),
        ),
    ],
    "shiyue_kyuketsuhime": [
        Annotation(
            "public_shiyue",
            "frame_03472.000.jpg",
            (735, 285, 1060, 585),
            ((755, 300, 955, 570),),
        ),
        Annotation(
            "public_shiyue",
            "frame_03474.500.jpg",
            (410, 285, 735, 590),
            ((425, 300, 610, 570),),
        ),
        Annotation(
            "public_shiyue",
            "frame_03475.500.jpg",
            (285, 285, 610, 590),
            ((300, 300, 490, 570),),
        ),
        Annotation(
            "public_shiyue",
            "frame_03476.500.jpg",
            (205, 285, 525, 590),
            ((220, 300, 410, 570),),
        ),
    ],
    "yaonian_enenra": [
        Annotation(
            "public_yaonian",
            "event_0198_00_05402.370.jpg",
            (940, 295, 1280, 590),
            ((965, 315, 1275, 570),),
        ),
        Annotation(
            "public_yaonian",
            "event_0198_02_05403.324.jpg",
            (735, 295, 1080, 590),
            ((760, 315, 1060, 570),),
        ),
        Annotation(
            "public_yaonian",
            "event_0245_00_06857.789.jpg",
            (570, 295, 930, 590),
            ((595, 315, 910, 490),),
        ),
    ],
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != (720, 1280):
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return image


def build_background(capture_dir: Path) -> np.ndarray:
    paths = sorted(capture_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No captures in {capture_dir}")

    # Sampling keeps memory bounded while still removing moving foreground objects.
    sample_paths = paths[:: max(1, len(paths) // 80)]
    stack = np.stack([read_image(path) for path in sample_paths])
    return np.median(stack, axis=0).astype(np.uint8)


def foreground_mask(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(image, background)
    mask = (np.max(difference, axis=2) >= 30).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backgrounds = {}
    for source, definition in SOURCES.items():
        background_path = definition["background"]
        backgrounds[source] = (
            read_image(background_path)
            if background_path is not None
            else build_background(definition["directory"])
        )
    manifest = {"version": 1, "threshold": 0.85, "templates": []}

    for name, annotations in ANNOTATIONS.items():
        for index, annotation in enumerate(annotations):
            source = SOURCES[annotation.source]
            filename = annotation.filename
            x1, y1, x2, y2 = annotation.bbox
            image = read_image(source["directory"] / filename)
            mask = foreground_mask(image, backgrounds[annotation.source])
            crop = image[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]

            if annotation.include:
                allowed = np.zeros_like(crop_mask)
                for ix1, iy1, ix2, iy2 in annotation.include:
                    allowed[max(0, iy1 - y1):iy2 - y1, max(0, ix1 - x1):ix2 - x1] = 255
                crop_mask = cv2.bitwise_and(crop_mask, allowed)

            # Ignore sparse background noise and the always-present player at the bottom.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(crop_mask)
            clean_mask = np.zeros_like(crop_mask)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] >= 80:
                    clean_mask[labels == component] = 255

            output_name = f"{name}_{index}.png"
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = clean_mask
            cv2.imwrite(str(OUTPUT_DIR / output_name), rgba)
            manifest["templates"].append(
                {
                    "name": name,
                    "file": output_name,
                    "source": f"{annotation.source}:{filename}",
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    print(f"Built {len(manifest['templates'])} templates in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
