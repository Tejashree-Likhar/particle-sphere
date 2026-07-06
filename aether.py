"""
Aether — Hand-Driven Particle Field (Python edition)
=====================================================

A 3000-particle 3D sphere rendered with pygame, driven live by your webcam
through MediaPipe's Hand Landmarker.

Gestures
--------
  One hand            -> move & rotate the sphere
  Pinch (thumb+index)  -> hold to charge (gold ring), release to burst
  Two hands            -> spread apart to grow, bring together to shrink
  Finger count (0-5)   -> shifts the sphere's color

Controls
--------
  Click "Webcam Background" (top right) or press [B]  -> toggle camera as background
  [Esc] or close window                                -> quit

Setup
-----
  pip install opencv-python mediapipe pygame numpy
  python aether.py

On first run the script downloads MediaPipe's hand_landmarker.task model
(~ a few MB) into the same folder. You need an internet connection once for
that; after that it runs fully offline.
"""

import os
import sys
import time
import math
import urllib.request

import numpy as np
import cv2
import pygame
import pygame.gfxdraw
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode
)

# ============================================================
# CONFIG
# ============================================================
PARTICLE_COUNT = 3000
BASE_RADIUS = 1.6                 # sphere radius in "world" units
WINDOW_SIZE = (1180, 760)
CAM_INDEX = 0
CAM_CAPTURE_SIZE = (640, 480)

BG_COLOR = (10, 11, 13)
INK = (234, 231, 224)
MUTED = (138, 141, 147)
GOLD = (212, 175, 106)

# finger-count -> color palette (curated jewel tones), RGB
FINGER_COLORS = [
    (154, 165, 177),   # 0 - fist (silver-blue)
    (94, 200, 216),    # 1 - cyan
    (124, 131, 240),   # 2 - periwinkle
    (224, 161, 92),    # 3 - amber
    (229, 107, 143),   # 4 - rose
    (111, 214, 160),   # 5 - emerald
]

CHARGE_TIME = 1.15     # seconds of pinch-hold to reach full charge
EXPLODE_DUR = 0.55
REFORM_DUR = 1.05

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

TIP = [8, 12, 16, 20]
PIP = [6, 10, 14, 18]


# ============================================================
# MODEL DOWNLOAD
# ============================================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-tracking model (one-time, ~8MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    except Exception as e:
        print(f"Could not download the model automatically ({e}).")
        print(f"Please download it manually from:\n  {MODEL_URL}")
        print(f"and place it at:\n  {MODEL_PATH}")
        sys.exit(1)


# ============================================================
# MATH HELPERS
# ============================================================
def fibonacci_sphere(n, radius):
    i = np.arange(n)
    offset = 2.0 / n
    increment = math.pi * (3.0 - math.sqrt(5.0))
    y = (i * offset - 1.0) + (offset / 2.0)
    r = np.sqrt(np.clip(1.0 - y * y, 0, None))
    phi = i * increment
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    pts = np.stack([x, y, z], axis=1) * radius
    return pts.astype(np.float32)


def rotation_matrix(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    return ry @ rx


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t):
    return 4 * t ** 3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def dist3(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def count_fingers(lm):
    """Count extended fingers from 21 hand landmarks.

    Orientation-independent: doesn't rely on the "Left"/"Right" handedness
    label (which can come back flipped depending on whether the frame fed to
    the detector was mirrored). The thumb is judged by how far it has moved
    away from the palm (toward the pinky side) rather than by absolute x
    direction, so it works the same for either hand.
    """
    c = 0
    if dist3(lm[4], lm[17]) > dist3(lm[3], lm[17]):
        c += 1
    for t, p in zip(TIP, PIP):
        if lm[t].y < lm[p].y:
            c += 1
    return c


def make_explosion_targets(base_positions, charge_amt, rng):
    n = base_positions.shape[0]
    bl = np.linalg.norm(base_positions, axis=1, keepdims=True)
    bl[bl == 0] = 1
    direction = base_positions / bl
    rand = (rng.random((n, 3)).astype(np.float32) * 2 - 1)
    rand[:, 0:2] *= 0.9
    rand[:, 2] *= 0.6
    combined = direction * 0.4 + rand
    cl = np.linalg.norm(combined, axis=1, keepdims=True)
    cl[cl == 0] = 1
    combined = combined / cl
    mag = BASE_RADIUS * (2.4 + charge_amt * 7.5) * (0.5 + rng.random((n, 1)).astype(np.float32) * 0.8)
    return combined * mag


# ============================================================
# APP
# ============================================================
class Aether:
    def __init__(self):
        ensure_model()

        # ---- mediapipe ----
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.6,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # ---- webcam ----
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CAPTURE_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CAPTURE_SIZE[1])
        if not self.cap.isOpened():
            print("Could not open webcam. Check that it's connected and not in use by another app.")
            sys.exit(1)

        # ---- pygame ----
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Aether — Hand-Driven Particle Field")
        self.clock = pygame.time.Clock()

        try:
            self.title_font = pygame.font.SysFont("Georgia,Times New Roman,Times", 42, italic=True)
        except Exception:
            self.title_font = pygame.font.SysFont(None, 42, italic=True)
        self.mono_small = pygame.font.SysFont("Consolas,Courier New,Monospace", 13)
        self.mono_tiny = pygame.font.SysFont("Consolas,Courier New,Monospace", 11)

        # ---- particle data ----
        self.base_positions = fibonacci_sphere(PARTICLE_COUNT, BASE_RADIUS)
        self.current_positions = self.base_positions.copy()
        self.explode_targets = np.zeros_like(self.base_positions)
        self.brightness = 0.55 + np.random.rand(PARTICLE_COUNT).astype(np.float32) * 0.55
        self.phase = np.random.rand(PARTICLE_COUNT).astype(np.float32) * math.pi * 2
        self.rng = np.random.default_rng()

        # ---- gesture / rig state ----
        self.num_hands = 0
        self.finger_count = -1
        self.hand_target = np.array([0.0, 0.0], dtype=np.float32)
        self.rig_pos = np.array([0.0, 0.0], dtype=np.float32)   # x,y offset (world units)
        self.rig_yaw = 0.0
        self.rig_pitch = 0.0
        self.target_scale = 1.0
        self.current_scale = 1.0
        self.pinching = False
        self.last_pinch = False
        self.charge = 0.0
        self.burst_state = "idle"   # idle | exploding | reforming
        self.burst_t = 0.0
        self.pinch_screen = None

        self.current_color = np.array(FINGER_COLORS[0], dtype=np.float32)
        self.target_color = np.array(FINGER_COLORS[0], dtype=np.float32)

        self.show_webcam_bg = False
        self.running = True

        # camera projection params
        self.cam_dist = 5.2
        self.focal = 900.0

        self.btn_rect = pygame.Rect(0, 0, 210, 34)  # positioned each frame (top-right)

    # ------------------------------------------------------
    def trigger_burst(self, charge_amt):
        self.explode_targets = make_explosion_targets(self.base_positions, charge_amt, self.rng)
        self.burst_state = "exploding"
        self.burst_t = 0.0

    # ------------------------------------------------------
    def process_hands(self, frame_rgb, dt):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        hands = result.hand_landmarks or []
        handed = result.handedness or []
        self.num_hands = len(hands)

        if self.num_hands == 2:
            c0, c1 = hands[0][9], hands[1][9]
            d = dist3(c0, c1)
            t = min(1.0, max(0.0, (d - 0.12) / (0.75 - 0.12)))
            self.target_scale = 0.5 + t * 1.9
            self.pinching = False
            self.last_pinch = False
            self.charge = max(0.0, self.charge - dt * 4)
            self.pinch_screen = None

        elif self.num_hands == 1:
            lm = hands[0]

            palm = lm[9]
            screen_x = 1 - palm.x   # mirror to match visual mirror
            screen_y = palm.y
            dx = (screen_x - 0.5) * 2
            dy = (0.5 - screen_y) * 2
            self.hand_target[:] = (dx, dy)

            fc = count_fingers(lm)
            if fc != self.finger_count:
                self.finger_count = fc
                self.target_color[:] = FINGER_COLORS[min(5, max(0, fc))]

            span = max(0.001, dist3(lm[0], lm[9]))
            pinch_dist = dist3(lm[4], lm[8]) / span
            pinching = pinch_dist < 0.42

            if pinching:
                self.charge = min(1.0, self.charge + dt / CHARGE_TIME)
            else:
                if self.last_pinch and self.charge > 0.04 and self.burst_state == "idle":
                    self.trigger_burst(self.charge)
                self.charge = max(0.0, self.charge - dt * 5)
            self.pinching = pinching
            self.last_pinch = pinching

            mid_x = 1 - (lm[4].x + lm[8].x) / 2
            mid_y = (lm[4].y + lm[8].y) / 2
            self.pinch_screen = (mid_x, mid_y)

        else:
            self.hand_target *= 0.98
            self.pinching = False
            self.last_pinch = False
            self.charge = max(0.0, self.charge - dt * 5)
            self.pinch_screen = None

    # ------------------------------------------------------
    def update(self, dt, now):
        ambient = 0.06
        self.rig_yaw += ambient * dt + self.hand_target[0] * 0.9 * dt
        self.rig_pitch += self.hand_target[1] * -0.6 * dt
        self.rig_pitch = max(-0.6, min(0.6, self.rig_pitch))

        target_pos = np.array([self.hand_target[0] * 1.1, self.hand_target[1] * 0.75], dtype=np.float32)
        alpha = 1 - pow(0.001, dt)
        self.rig_pos += (target_pos - self.rig_pos) * alpha

        self.current_scale += (self.target_scale - self.current_scale) * min(1.0, dt * 4)

        self.current_color += (self.target_color - self.current_color) * min(1.0, dt * 4)

        if self.burst_state == "exploding":
            self.burst_t += dt / EXPLODE_DUR
            t = ease_out_cubic(min(1.0, self.burst_t))
            self.current_positions = self.base_positions + (self.explode_targets - self.base_positions) * t
            if self.burst_t >= 1.0:
                self.burst_state = "reforming"
                self.burst_t = 0.0
        elif self.burst_state == "reforming":
            self.burst_t += dt / REFORM_DUR
            t = ease_in_out_cubic(min(1.0, self.burst_t))
            self.current_positions = self.explode_targets + (self.base_positions - self.explode_targets) * t
            if self.burst_t >= 1.0:
                self.burst_state = "idle"
        else:
            bl = np.linalg.norm(self.base_positions, axis=1, keepdims=True)
            bl[bl == 0] = 1
            direction = self.base_positions / bl
            w = 0.02 * np.sin(now * 1.6 + self.phase)
            self.current_positions = self.base_positions + direction * w[:, None]

    # ------------------------------------------------------
    def project(self):
        R = rotation_matrix(self.rig_yaw, self.rig_pitch)
        world = (self.current_positions @ R.T) * self.current_scale
        world[:, 0] += self.rig_pos[0]
        world[:, 1] += self.rig_pos[1]

        w, h = self.screen.get_size()
        cx, cy = w / 2, h / 2

        z = self.cam_dist - world[:, 2]
        z = np.clip(z, 0.05, None)
        sx = self.focal * world[:, 0] / z + cx
        sy = -self.focal * world[:, 1] / z + cy
        size = np.clip((self.focal * 0.028) / z, 0.6, 7.0)
        depth_dim = np.clip(3.0 / z, 0.5, 1.4)

        visible = (sx > -10) & (sx < w + 10) & (sy > -10) & (sy < h + 10)
        return sx[visible], sy[visible], size[visible], depth_dim[visible], self.brightness[visible]

    # ------------------------------------------------------
    def draw_particles(self):
        sx, sy, size, depth_dim, bright = self.project()
        r, g, b = self.current_color
        factor = (bright * depth_dim)
        cols_r = np.clip(r * factor, 0, 255).astype(np.int32)
        cols_g = np.clip(g * factor, 0, 255).astype(np.int32)
        cols_b = np.clip(b * factor, 0, 255).astype(np.int32)

        sx_i = sx.astype(np.int32).tolist()
        sy_i = sy.astype(np.int32).tolist()
        size_i = np.clip(size, 1, None).astype(np.int32).tolist()
        cr = cols_r.tolist()
        cg = cols_g.tolist()
        cb = cols_b.tolist()

        draw_circle = pygame.gfxdraw.filled_circle
        surface = self.screen
        for i in range(len(sx_i)):
            draw_circle(surface, sx_i[i], sy_i[i], size_i[i], (cr[i], cg[i], cb[i]))

    # ------------------------------------------------------
    def draw_webcam_background(self, frame_bgr):
        w, h = self.screen.get_size()
        frame = cv2.resize(frame_bgr, (w, h))
        frame = cv2.flip(frame, 1)                    # mirror horizontally
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pygame.surfarray.make_surface wants shape (width, height, 3), i.e.
        # array[x, y] = pixel. cv2 frames are (height, width, 3), i.e.
        # frame[y, x] = pixel. Swap the first two axes (a transpose) rather
        # than rotate — np.rot90 here would spin the image 90 degrees, which
        # is what caused the upside-down/sideways background before.
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0, 0))
        dark = pygame.Surface((w, h), pygame.SRCALPHA)
        dark.fill((10, 11, 13, 150))
        self.screen.blit(dark, (0, 0))

    # ------------------------------------------------------
    def draw_ui(self):
        w, h = self.screen.get_size()

        # corner brackets
        L = 24
        m = 22
        bcol = (INK[0], INK[1], INK[2])
        for (ox, oy, dx, dy) in [(m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)]:
            pygame.draw.line(self.screen, bcol, (ox, oy), (ox + dx * L, oy), 1)
            pygame.draw.line(self.screen, bcol, (ox, oy), (ox, oy + dy * L), 1)

        # title
        title_surf = self.title_font.render("Aether", True, INK)
        self.screen.blit(title_surf, (44, 34))
        sub_surf = self.mono_tiny.render("PARTICLE FIELD  —  HAND INTERFACE", True, MUTED)
        self.screen.blit(sub_surf, (46, 34 + title_surf.get_height() + 4))

        # webcam toggle button
        self.btn_rect = pygame.Rect(w - 44 - 210, 40, 210, 34)
        pygame.draw.rect(self.screen, (234, 231, 224, 40), self.btn_rect, 1)
        dot_col = GOLD if self.show_webcam_bg else MUTED
        pygame.draw.circle(self.screen, dot_col, (self.btn_rect.x + 16, self.btn_rect.centery), 4)
        btn_label = self.mono_tiny.render("WEBCAM BACKGROUND  [B]", True, INK)
        self.screen.blit(btn_label, (self.btn_rect.x + 28, self.btn_rect.centery - 6))

        # legend
        legend_lines = [
            ("ONE HAND", "move & rotate the field"),
            ("PINCH + HOLD", "charge, release to burst"),
            ("TWO HANDS", "spread to grow, close to shrink"),
            ("FINGERS 0-5", "shift the field's color"),
        ]
        ly = h - 40 - len(legend_lines) * 20
        for key, desc in legend_lines:
            key_surf = self.mono_tiny.render(f"{key:<14}", True, (INK[0], INK[1], INK[2], 160))
            desc_surf = self.mono_tiny.render(desc, True, MUTED)
            self.screen.blit(key_surf, (44, ly))
            self.screen.blit(desc_surf, (44 + 130, ly))
            ly += 20

        # HUD (bottom-right)
        gesture = "idle"
        if self.burst_state == "exploding":
            gesture = "burst"
        elif self.burst_state == "reforming":
            gesture = "reforming"
        elif self.num_hands == 2:
            gesture = "scaling"
        elif self.pinching:
            gesture = "charging"
        elif self.num_hands == 1:
            gesture = "tracking"

        fingers_txt = str(self.finger_count) if self.num_hands == 1 else "-"
        hud_lines = [
            f"HANDS {self.num_hands}",
            f"FINGERS {fingers_txt}",
            f"GESTURE {gesture}",
        ]
        hy = h - 40 - len(hud_lines) * 20
        for line in hud_lines:
            surf = self.mono_tiny.render(line, True, MUTED)
            self.screen.blit(surf, (w - 44 - surf.get_width(), hy))
            hy += 20
        swatch_col = tuple(int(c) for c in self.current_color)
        pygame.draw.circle(self.screen, swatch_col, (w - 44 - 8, hy - 20 + 6), 5)

        # pinch charge ring
        if self.pinch_screen and (self.pinching or self.charge > 0.02):
            px = int(self.pinch_screen[0] * w)
            py = int(self.pinch_screen[1] * h)
            radius = 34
            rect = pygame.Rect(px - radius, py - radius, radius * 2, radius * 2)
            pygame.draw.circle(self.screen, (INK[0], INK[1], INK[2]), (px, py), radius, 1)
            end_angle = -math.pi / 2 + self.charge * 2 * math.pi
            if self.charge > 0.005:
                pygame.draw.arc(self.screen, GOLD, rect, -math.pi / 2, end_angle, 3)
            pct = self.mono_tiny.render(f"{int(self.charge*100)}%", True, GOLD)
            self.screen.blit(pct, (px - pct.get_width() // 2, py - pct.get_height() // 2))

    # ------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_b:
                    self.show_webcam_bg = not self.show_webcam_bg
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.btn_rect.collidepoint(event.pos):
                    self.show_webcam_bg = not self.show_webcam_bg
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

    # ------------------------------------------------------
    def run(self):
        prev = time.time()
        while self.running:
            self.handle_events()

            ok, frame_bgr = self.cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            now = time.time()
            dt = min(0.05, now - prev)
            prev = now

            self.process_hands(frame_rgb, dt)
            self.update(dt, now)

            if self.show_webcam_bg:
                self.draw_webcam_background(frame_bgr)
            else:
                self.screen.fill(BG_COLOR)

            self.draw_particles()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)

        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    Aether().run()
