# app.py  - online / phone version (uses the browser camera via WebRTC)
import json
import time

import cv2
import requests

import av
import streamlit.components.v1 as components
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from detector import initialize_face_mesh, process_frame
from fatigue import analyze_landmarks
from driver_utils import draw_overlay

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


def _tg_config():
    """Read Telegram settings from Streamlit Secrets (never store them in GitHub)."""
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except Exception:
        return None, None


def send_telegram(text, jpg=None):
    """Send a message (and optional photo) to the owner. Returns (ok, info)."""
    token, chat_id = _tg_config()
    if not token or not chat_id:
        return False, "Telegram is not set up yet (add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Secrets)."
    base = f"https://api.telegram.org/bot{token}"
    try:
        if jpg is not None:
            r = requests.post(f"{base}/sendPhoto", data={"chat_id": chat_id, "caption": text},
                              files={"photo": ("driver.jpg", jpg, "image/jpeg")}, timeout=10)
        else:
            r = requests.post(f"{base}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=10)
        data = r.json()
        if data.get("ok"):
            return True, "Sent"
        return False, data.get("description", "Telegram refused the message")
    except Exception as e:
        return False, f"Could not reach Telegram ({type(e).__name__})"


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
        self.reason = ""
        self.last_img = None
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
        self.bad_since = None
        self.ok_since = None
        self.bad_seconds = 0.0

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

        reason = ""
        if status != "Normal":
            if gaze == "Down (Hidden)":
                reason = "Face not visible"
            elif self.closed_frames > 10:
                reason = "Eyes closed"
            elif self.mouth_open_frames > 4:
                reason = "Yawning"
            elif self.head_down_frames > 4:
                reason = "Head down"
            else:
                reason = "Drowsy"

        self.status, self.fatigue, self.gaze, self.reason = status, fatigue, gaze, reason

        # how long has the driver been CONTINUOUSLY inattentive?
        # (a short normal moment of up to 3 s does not reset the timer)
        t = time.time()
        if status != "Normal":
            self.ok_since = None
            if self.bad_since is None:
                self.bad_since = t
        else:
            if self.ok_since is None:
                self.ok_since = t
            if self.bad_since is not None and t - self.ok_since > 3.0:
                self.bad_since = None
        self.bad_seconds = (t - self.bad_since) if self.bad_since else 0.0

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
        self.last_img = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")


VOICE = {
    ("Danger", "Eyes closed"): "Danger! Your eyes are closed. Wake up!",
    ("Danger", "Yawning"): "Danger! You are very sleepy. Please take a break.",
    ("Danger", "Head down"): "Danger! Keep your head up and look at the road!",
    ("Danger", "Face not visible"): "Danger! I cannot see your face. Look at the road!",
    ("Warning", "Eyes closed"): "Warning. Keep your eyes open.",
    ("Warning", "Yawning"): "Warning. You seem tired.",
    ("Warning", "Head down"): "Warning. Please look up.",
}

ALERT_JS = """<script>/*__NONCE__*/
// 1) beep
if (__BEEP__) { try {
  const C = window.AudioContext || window.webkitAudioContext;
  const ctx = new C(); ctx.resume();
  for (let i = 0; i < __BEEPS__; i++) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'square'; o.frequency.value = __FREQ__; g.gain.value = 0.5;
    o.connect(g); g.connect(ctx.destination);
    const t = ctx.currentTime + i * 0.35; o.start(t); o.stop(t + 0.25);
  }
} catch (e) {} }
// 2) full-screen flash
if (__FLASH__) { try {
  const d = window.parent.document;
  if (!d.getElementById('dms-style')) {
    const st = d.createElement('style'); st.id = 'dms-style';
    st.textContent = '@keyframes dmsflash{0%{opacity:0}12%{opacity:.8}30%{opacity:.15}50%{opacity:.75}100%{opacity:0}}';
    d.head.appendChild(st);
  }
  const old = d.getElementById('dms-flash'); if (old) old.remove();
  const el = d.createElement('div'); el.id = 'dms-flash';
  el.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none;opacity:0;background:__COLOR__;animation:dmsflash 1.3s ease-out forwards;';
  d.body.appendChild(el);
} catch (e) {} }
// 3) voice warning (uses the main page's speech engine so it survives iframe refresh)
if (__VOICE__) { try {
  const W = window.parent, S = W.speechSynthesis, now = Date.now();
  // speak if idle, or if the engine looks stuck (more than 5 s since last start)
  if (S && (!S.speaking || now - (W.__dmsVoiceAt || 0) > 5000)) {
    S.cancel(); S.resume();
    const u = new W.SpeechSynthesisUtterance(__TEXT__);
    u.rate = 1.0; u.volume = 1; u.lang = 'en-US';
    W.__dmsU = u; W.__dmsVoiceAt = now;      // keep a reference so it is not garbage-collected
    W.setTimeout(function () { S.speak(u); }, 120);
  }
} catch (e) {} }
// 4) vibration
if (__VIBRATE__) { try { navigator.vibrate(__VIB__); } catch (e) {} }
</script>"""


def fire_alert(box, status, reason, beep=True, voice=True, flash=True, vibrate=True):
    """Phone alert: beep + spoken warning + full-screen flash + vibration."""
    danger = status == "Danger"
    text = VOICE.get((status, reason),
                     "Danger! Stay alert!" if danger else "Warning. You look drowsy.")
    js = (ALERT_JS
          .replace("__NONCE__", str(time.time_ns()))
          .replace("__BEEP__", "true" if beep else "false")
          .replace("__BEEPS__", "2" if danger else "1")
          .replace("__FREQ__", "2200" if danger else "1200")
          .replace("__FLASH__", "true" if flash else "false")
          .replace("__COLOR__", "rgba(255,0,0,1)" if danger else "rgba(255,170,0,1)")
          .replace("__VOICE__", "true" if voice else "false")
          .replace("__TEXT__", json.dumps(text))
          .replace("__VIBRATE__", "true" if vibrate else "false")
          .replace("__VIB__", "[300,100,300,100,300]" if danger else "[250]"))
    with box:
        components.html(js, height=0)


st.caption("Tap START, allow camera access, and place your phone so your face is visible.")

status_box = st.empty()
notify_box = st.empty()

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

with st.expander("⚙️ Alert settings", expanded=False):
    opt_beep = st.toggle("🔔 Beep", value=True)
    opt_voice = st.toggle("🗣️ Voice warning", value=True)
    opt_flash = st.toggle("🔴 Full-screen flash", value=True)
    opt_vibrate = st.toggle("📳 Vibration", value=True)

with st.expander("📨 Owner notification (Telegram)", expanded=False):
    notify_on = st.toggle("Message the owner if driver stays inattentive", value=True)
    notify_after = st.number_input("Send after (seconds)", min_value=10, max_value=600, value=60, step=5)
    driver_name = st.text_input("Driver / vehicle name", value="My vehicle")
    if st.button("📨 Send test message"):
        ok, info = send_telegram(f"✅ Test message from AI Driver Monitor ({driver_name}). Notifications are working.")
        st.success("Test message sent!") if ok else st.error(info)

b1, b2 = st.columns(2)
if b1.button("🔄 Reset counters") and ctx.video_processor:
    ctx.video_processor.reset()
if b2.button("🔊 Test alert"):
    # tapping a button also lets the browser allow sound and voice
    fire_alert(test_box, "Danger", "Eyes closed", opt_beep, opt_voice, opt_flash, opt_vibrate)


def show_status(status: str, reason: str = ""):
    why = f": {reason.upper()}" if reason else ""
    if status == "Danger":
        status_box.markdown(f'<div class="status-card status-danger">🚨 DANGER{why}</div>', unsafe_allow_html=True)
    elif status == "Warning":
        status_box.markdown(f'<div class="status-card status-warning">⚠️ WARNING{why}</div>', unsafe_allow_html=True)
    else:
        status_box.markdown('<div class="status-card status-normal">✅ STATUS: ATTENTIVE</div>', unsafe_allow_html=True)


if ctx.state.playing:
    last_beep = 0.0
    while ctx.state.playing:
        p = ctx.video_processor
        if p is not None:
            show_status(p.status, p.reason)
            fatigue_box.metric("Fatigue Index", f"{p.fatigue:.2f}")
            gaze_box.metric("Gaze", p.gaze)
            blink_box.metric("Blinks", str(p.blinks))
            yawn_box.metric("Yawns", str(p.yawns))

            now = time.time()

            # ---- owner notification after continuous inattention
            if notify_on:
                if p.bad_seconds > 0:
                    notify_box.progress(min(p.bad_seconds / notify_after, 1.0),
                                        text=f"Inattentive for {int(p.bad_seconds)}s / {int(notify_after)}s")
                else:
                    notify_box.empty()
                if p.bad_seconds >= notify_after and not st.session_state.get("notified") \
                        and now - st.session_state.get("last_sent", 0) > 120:
                    jpg = None
                    if p.last_img is not None:
                        okj, buf = cv2.imencode(".jpg", p.last_img)
                        jpg = buf.tobytes() if okj else None
                    msg = (f"🚨 DRIVER ALERT - {driver_name}\n"
                           f"The driver has been inattentive for {int(p.bad_seconds)} seconds.\n"
                           f"Reason: {p.reason or 'Drowsy'}\n"
                           f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n"
                           f"Please call the driver immediately.")
                    ok, info = send_telegram(msg, jpg)
                    st.session_state["notified"] = True
                    st.session_state["last_sent"] = now
                    st.toast("📨 Owner notified" if ok else f"⚠️ Notification failed: {info}")
                if p.bad_seconds == 0:
                    st.session_state["notified"] = False

            if opt_voice:
                gap = 2.5 if p.status == "Danger" else 4.0
            else:
                gap = 1.4 if p.status == "Danger" else 2.0
            if p.status in ("Warning", "Danger") and now - last_beep > gap:
                fire_alert(sound_box, p.status, p.reason, opt_beep, opt_voice, opt_flash, opt_vibrate)
                last_beep = now
            elif p.status == "Normal" and now - last_beep > 2.0:
                sound_box.empty()
        time.sleep(0.3)
else:
    show_status("Normal")
    status_box.markdown('<div class="status-card status-normal">ℹ️ System Offline - tap START</div>', unsafe_allow_html=True)
