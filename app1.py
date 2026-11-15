    # app.py  - online / phone version (uses the browser camera via WebRTC)
import threading
import time

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from detector import initialize_face_mesh, process_frame
from fatigue import analyze_landmarks
from driver_utils import DANGER_BEEP, WARNING_BEEP, draw_overlay

st.set_page_config(
    page_title="AI Driver Monitoring System",
    page_icon="🚘",
    layout="centered",
)

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 50% 50%, #0d1117 0%, #030712 100%); color: #f3f4f6; }
.main-title { font-size: 1.6rem; font-weight: 800; text-align: center; margin-bottom: 14px;
  background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
div[data-testid="stMetric"] { background: rgba(17,24,39,0.75); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 10px 14px; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800 !important; color: #38bdf8 !important; }
div[data-testid="stMetricLabel"] { color: #9ca3af !important; text-transform: uppercase; }
.status-card { padding: 14px; border-radius: 14px; text-align: center; font-size: 1.1rem;
  font-weight: 800; margin-bottom: 12px; }
.status-normal { background: rgba(6,78,59,0.6); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.status-warning { background: rgba(120,53,15,0.7); color: #fbbf24; border: 1px solid rgba(251,191,36,0.5); }
.status-danger { background: rgba(127,29,29,0.8); color: #f87171; border: 1px solid rgba(248,113,113,0.6); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚘 AI DRIVER MONITORING SYSTEM</div>', unsafe_allow_html=True)


class DriverProcessor(VideoProcessorBase):
    """Runs on camera frames from the browser. Analysis is limited to ~10 times
    per second so the video stays smooth, and counters are time-based so the
    fatigue score does not depend on how fast the server is."""

    def __init__(self):
        self.face_mesh = initialize_face_mesh()
        self.reset()
        self.status = "Normal"
        self.fatigue = 0.0
        self.gaze = "Center"
        self._last_proc = 0.0
        self._prev = time.time()
        self.fps = 0.0

    def reset(self):
        self.blinks = 0
        self.yawns = 0
        self.closed_frames = 0.0
        self.mouth_open_frames = 0.0
        self.head_down_frames = 0.0
        self.was_eye_closed = False
        self.was_yawning = False

    def _analyze(self, img, step):
        img, results = process_frame(img, self.face_mesh)
        status, fatigue, gaze = "Normal", 0.0, "Center"

        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            (status, fatigue, gaze, self.closed_frames, self.mouth_open_frames,
             self.head_down_frames, eye_closed, yawning) = analyze_landmarks(
                lm, self.closed_frames, self.mouth_open_frames,
                self.head_down_frames, step)

            if self.was_eye_closed and not eye_closed:
                self.blinks += 1
            self.was_eye_closed = eye_closed
            if self.was_yawning and not yawning:
                self.yawns += 1
            self.was_yawning = yawning
        else:
            # face not visible (head far down / turned away)
            self.head_down_frames += step
            if self.head_down_frames > 10:
                status = "Danger"
                fatigue = min(1.0, 0.40 + self.head_down_frames * 0.05)
                gaze = "Down (Hidden)"

        self.status, self.fatigue, self.gaze = status, fatigue, gaze

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # video fps (smoothed)
        inst = 1 / (now - self._prev) if now > self._prev else 0
        self.fps = 0.8 * self.fps + 0.2 * inst
        self._prev = now

        if now - self._last_proc >= 0.1:
            dt = (now - self._last_proc) if self._last_proc else 0.1
            self._last_proc = now
            step = min(dt * 30.0, 6.0)   # "frames at 30 fps" that passed
            self._analyze(img, step)

        img = draw_overlay(img, self.status, self.fps)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.caption("Tap START, allow camera access, and place your phone so your face is visible.")

status_box = st.empty()

ctx = webrtc_streamer(
    key="driver-monitor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DriverProcessor,
    media_stream_constraints={
        "video": {"facingMode": "user", "width": {"ideal": 320}, "height": {"ideal": 240},
                  "frameRate": {"ideal": 15, "max": 20}},
        "audio": False,
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

c1, c2 = st.columns(2)
fatigue_box = c1.empty()
gaze_box = c2.empty()
c3, c4 = st.columns(2)
blink_box = c3.empty()
yawn_box = c4.empty()
sound_box = st.empty()
test_box = st.empty()

b1, b2 = st.columns(2)
if b1.button("🔄 Reset counters") and ctx.video_processor:
    ctx.video_processor.reset()
if b2.button("🔊 Test sound"):
    # tapping a button lets the browser allow sound; also checks your volume
    with test_box:
        st.audio(DANGER_BEEP, format="audio/wav", autoplay=True)
sound_on = st.toggle("🔔 Sound alerts", value=True)


def show_status(status: str):
    if status == "Danger":
        status_box.markdown('<div class="status-card status-danger">🚨 DANGER: UNATTENTIVE</div>', unsafe_allow_html=True)
    elif status == "Warning":
        status_box.markdown('<div class="status-card status-warning">⚠️ WARNING: DROWSY</div>', unsafe_allow_html=True)
    else:
        status_box.markdown('<div class="status-card status-normal">✅ STATUS: ATTENTIVE</div>', unsafe_allow_html=True)


if ctx.state.playing:
    last_beep = 0.0
    beep_count = 0
    while ctx.state.playing:
        p = ctx.video_processor
        if p is not None:
            show_status(p.status)
            fatigue_box.metric("Fatigue Index", f"{p.fatigue:.2f}")
            gaze_box.metric("Gaze", p.gaze)
            blink_box.metric("Blinks", str(p.blinks))
            yawn_box.metric("Yawns", str(p.yawns))

            now = time.time()
            gap = 0.9 if p.status == "Danger" else 1.8
            if sound_on and p.status in ("Warning", "Danger") and now - last_beep > gap:
                beep_count += 1
                data = DANGER_BEEP if p.status == "Danger" else WARNING_BEEP
                # tiny change in the bytes makes the browser replay the sound
                data = data + bytes(beep_count % 7)
                with sound_box:
                    st.audio(data, format="audio/wav", autoplay=True)
                last_beep = now
            elif p.status == "Normal" and now - last_beep > 2.0:
                sound_box.empty()
        time.sleep(0.3)
else:
    show_status("Normal")
    status_box.markdown('<div class="status-card status-normal">ℹ️ System Offline - tap START</div>', unsafe_allow_html=True)
        
