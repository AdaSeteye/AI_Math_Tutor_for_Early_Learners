"""
Child-facing Gradio demo: on open, Kinyarwanda welcome + question TTS, colourful count image, tap or voice.
Run: pip install -r requirements.txt && python demo.py

Design: auto-start; feedback in the session language (Kinyarwanda by default, or ASR-detected);
auto-advance after each answer (star animation on correct, then next image + TTS);
10s silence = question repeat + pad highlight; 20s = simpler item. No on-image text; no mentor readouts.
"""

from __future__ import annotations

import os
import socket
import tempfile
import time
import wave
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from tutor.curriculum_loader import LanguageCode
from tutor.feedback_audio import synthesize_text_to_wav
from tutor.lang_detect import detect_child_utterance
from tutor.pipeline import TutorSession
from tutor.visuals import render_count_image

# ASR for mic path: default on so 7-9 (voice-primary) works without extra env.
MIC_ASR = os.environ.get("TUTOR_ENABLE_MIC_ASR", "1").lower() in ("1", "true", "yes", "on")

# Warm Kinyarwanda welcome; combined with the item prompt in TTS (no on-screen text).
WELCOME_KIN = "Murakaza neza! Tugabane guhabura! "


def _load_session() -> TutorSession:
    s = TutorSession.from_default(language="kin")
    s.reset_to_demo_start()
    return s


def _b2pil(data: bytes | None) -> Image.Image | None:
    if data is None:
        return None
    return Image.open(BytesIO(data))


def _question_tts_path(text: str, lang: LanguageCode) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = Path(f.name)
    if synthesize_text_to_wav(t, lang, p, kind="encourage"):
        return str(p)
    p.unlink(missing_ok=True)
    return None


def do_present(
    session: TutorSession,
    *,
    include_welcome: bool = False,
) -> tuple[Image.Image | None, str, TutorSession, str | None, str | None]:
    """Return image, task placeholder, session, play wav, and a question-only wav for 10s repeat."""
    s = session
    pr = s.present()
    if pr is None:
        return None, "", s, None, None
    img_data: bytes | None = pr.image_png
    if img_data is None and pr.item.visual and pr.item.type == "count_image":
        v = pr.item.visual
        n = int(v.get("n_objects", 0))
        lab = str(v.get("object_label", "circle"))
        img_data = render_count_image(
            n, object_label=lab, seed=hash(pr.item.id) % (2**31), with_caption=False
        )
    img = _b2pil(img_data)
    prompt = (pr.prompt or "").strip()
    rep = _question_tts_path(prompt, pr.language)
    if include_welcome and prompt:
        long_text = f"{WELCOME_KIN}{prompt}"
        main = _question_tts_path(long_text, pr.language) or rep
    else:
        main = rep
    return img, "", s, main, rep


def do_score_tap(
    session: TutorSession,
    value: int,
) -> tuple[str, str | None, TutorSession, bool]:
    return _score_and_format(session, int(value), "tap")


def do_score_from_mic(
    session: TutorSession,
    audio: tuple | None,
) -> tuple[str, str | None, TutorSession, bool | None]:
    if not MIC_ASR or audio is None:
        e = _empty_result(session)
        return e[0], e[1], e[2], None
    sr, data = audio[0], audio[1]
    if data is None or (hasattr(data, "size") and data.size == 0):
        e = _empty_result(session)
        return e[0], e[1], e[2], None
    wlang = "rw" if session.language == "kin" else str(session.language)
    try:
        if (os.environ.get("TUTOR_ASR_ENGINE", "whisper") or "whisper").lower() == "mms" and os.environ.get(
            "TUTOR_MMS_ASR", ""
        ).lower() in ("1", "true", "on", "yes"):
            from tutor.asr_adapt import transcribe_mms_1b_all

            tx = transcribe_mms_1b_all(
                np.asarray(data, dtype=np.float32), int(sr), language=wlang
            )
        else:
            from tutor.asr_adapt import transcribe_and_detect

            tx, _ = transcribe_and_detect(
                np.asarray(data, dtype=np.float32), int(sr), language=None
            )
            tx = (tx or "").strip()
    except OSError:
        e = _empty_result(session)
        return e[0], e[1], e[2], None
    if not (tx or "").strip():
        e = _empty_result(session)
        return e[0], e[1], e[2], None
    return do_score_voice(session, tx)


def _empty_result(session: TutorSession) -> tuple[str, str | None, TutorSession]:
    return '<div class="child-result" aria-hidden="true"></div>', None, session


def do_score_voice(
    session: TutorSession,
    transcript: str,
) -> tuple[str, str | None, TutorSession, bool]:
    t = (transcript or "").strip()
    if not t:
        return _score_and_format(session, None, "voice")
    prof = detect_child_utterance(t)
    session.set_language(prof.dominant)
    r = session.score(
        t,
        "voice",
        feedback_lang=prof.dominant,
        feedback_extra_tts=prof.l2_numeral_appendix.strip() or None,
        child_language_note=prof.child_summary,
    )
    return _result_visual(r.correct), r.feedback_wav_path, session, r.correct


def _result_visual(correct: bool) -> str:
    icon = "🌟" if correct else "💪"
    cls = "child-yay" if correct else "child-try"
    ex = " star-anim" if correct else ""
    return (
        f'<div class="child-result {cls}{ex}" role="img" aria-label="feedback">'
        f'<span class="big-emoji" aria-hidden="true">{icon}</span></div>'
    )


def _concat_wav_files(fb: str | None, nq: str | None, *, pause_s: float = 0.35) -> str | None:
    """Play feedback, short silence, then next question in one file when formats match."""
    if not nq and not fb:
        return None
    if not nq:
        return fb
    if not fb:
        return nq
    try:
        with wave.open(fb) as a, wave.open(nq) as b:
            ap, bp = a.getparams(), b.getparams()
            if ap[:3] != bp[:3]:
                return nq
            f1, f2 = a.readframes(a.getnframes()), b.readframes(b.getnframes())
        nch, sw, fr = ap.nchannels, ap.sampwidth, ap.framerate
        npad = int(float(fr) * pause_s) * int(sw) * int(nch)
        pad = b"\x00" * npad
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            outp = Path(t.name)
        with wave.open(str(outp), "w") as w:
            w.setparams(ap)
            w.writeframes(f1 + pad + f2)
        return str(outp)
    except (OSError, ValueError):
        return nq


def _score_and_format(
    session: TutorSession,
    response: int | str | None,
    mode: str,
) -> tuple[str, str | None, TutorSession, bool]:
    if session.current_item() is None:
        return _empty_result_html(), None, session, False
    r = session.score(response, mode="tap" if mode == "tap" else "voice")
    return _result_visual(r.correct), r.feedback_wav_path, session, r.correct


def do_next(
    session: TutorSession,
) -> tuple[Image.Image | None, str, str | None, TutorSession, str | None, str | None]:
    session.advance()
    pr = session.present()
    if pr is None:
        return None, _done_result_html(), None, session, None, None
    img_data: bytes | None = pr.image_png
    if img_data is None and pr.item.visual and pr.item.type == "count_image":
        v = pr.item.visual
        n = int(v.get("n_objects", 0))
        lab = str(v.get("object_label", "circle"))
        img_data = render_count_image(
            n, object_label=lab, seed=hash(pr.item.id) % (2**31), with_caption=False
        )
    img = _b2pil(img_data)
    ptext = (pr.prompt or "").strip()
    rep = _question_tts_path(ptext, pr.language)
    return img, _empty_result_html(), rep, session, rep, rep


def _empty_result_html() -> str:
    return '<div class="child-result" aria-hidden="true"></div>'


def _done_result_html() -> str:
    return '<div class="child-result" role="img" aria-label="done">🎉</div>'


def build_ui() -> tuple[gr.Blocks, object, str]:
    child_css = """
    .play-stack { display: flex; flex-direction: column; }
    .ux-tap .play-stack { flex-direction: column-reverse; }
    .ux-tap .num-pad { min-height: 3.05rem !important; min-width: 2.4rem; font-size: 1.35rem; }
    .ux-tap .mic-pri { min-height: 2.65rem; min-width: 3.2rem; }
    .ux-voice .mic-pri { min-height: 3.55rem !important; min-width: 3.5rem; font-size: 1.45rem; }
    .ux-voice .num-pad { min-height: 2.5rem !important; font-size: 1.05rem; }
    .child-pick { font-size: 2.5rem; text-align: center; font-weight: 800; min-height: 1.1em; color: #e65100; }
    .gr-form .num-pad { min-height: 2.75rem !important; min-width: 2.3rem; font-size: 1.1rem; font-weight: 800; border-radius: 14px; }
    .num-grid { display: flex; flex-wrap: wrap; gap: 0.35rem; justify-content: center; max-width: 32rem; margin: 0 auto; }
    .pad-glow .num-pad { box-shadow: 0 0 0 3px #ffb300; border-color: #ff8f00; }
    @keyframes softPulse { 0%,100% { filter: brightness(1);} 50% { filter: brightness(1.12);} }
    .pad-glow { animation: softPulse 1.2s ease-in-out infinite; }
    .child-result { text-align: center; padding: 0.4rem; min-height: 2rem; }
    .child-result .big-emoji { font-size: 3rem; line-height: 1; display: block; }
    @keyframes child-pop { 0% { transform: scale(0.5) rotate(-18deg);} 50% { transform: scale(1.2) rotate(8deg);} 100% { transform: scale(1) rotate(0deg);} }
    @keyframes star-twirl { 0% { filter: drop-shadow(0 0 6px #ffc107) brightness(1);} 100% { filter: drop-shadow(0 0 16px #ffab00) brightness(1.15);} }
    @keyframes star-sparkle { 0%, 100% { transform: scale(1); opacity: 1;} 30% { transform: scale(1.35) rotate(12deg);} 60% { transform: scale(1.15) rotate(-6deg);} }
    .child-yay .big-emoji { animation: child-pop 0.55s ease-out, star-twirl 0.7s ease-out, star-sparkle 0.8s ease-in-out; }
    .child-result.star-anim { position: relative; }
    .bar-btn { min-height: 2.85rem; min-width: 3.2rem; font-size: 1.25rem; font-weight: 800; }
    .sr-only-parent { position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden; }
    """

    try:
        theme = gr.themes.Soft(
            font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "sans-serif"],
            primary_hue="amber",
        )
    except Exception:
        theme = gr.themes.Soft()

    def _n_noop(n: int) -> list:
        return [gr.update() for _ in range(n)]

    with gr.Blocks(title="Math play") as demo:
        with gr.Column(elem_classes=["ux", "ux-tap"]) as main_ux:
            session_state = gr.State(_load_session())
            num_picked = gr.State(0)
            awaiting = gr.State(True)
            stim_t0 = gr.State(-1.0)
            fired_10 = gr.State(False)
            did_20_once = gr.State(False)
            last_repeat_wav = gr.State(None)

            age = gr.Radio(
                choices=[("🧒", "56"), ("🧑", "79")],
                value="56",
                show_label=False,
                visible=False,
            )

            image = gr.Image(type="pil", height=300, show_label=False)
            pick_show = gr.HTML(value='<p class="child-pick" aria-label="number">?</p>')

            with gr.Column(elem_classes=["play-stack"]):
                with gr.Row():
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        show_label=False,
                        elem_classes=["mic-pri"],
                    )
                with gr.Column(elem_classes=["num-grid", "num-pad-outer"]) as num_pad_wrap:
                    num_rows = (list(range(0, 6)), list(range(6, 12)), list(range(12, 18)), [18, 19, 20])
                    num_btns: list[gr.Button] = []
                    for rnums in num_rows:
                        with gr.Row():
                            for n in rnums:
                                b = gr.Button(
                                    str(n),
                                    scale=0,
                                    elem_classes=["num-pad"],
                                )
                                num_btns.append(b)

            def on_pick(n: int) -> tuple[int, str]:
                return n, f'<p class="child-pick" aria-label="number">{n}</p>'

            for n, b in zip(range(21), num_btns):
                b.click(lambda x=n: on_pick(x), outputs=[num_picked, pick_show])
            with gr.Row(equal_height=True):
                check_tap = gr.Button("✅", elem_classes=["bar-btn"], min_width=90)
                check_mic = gr.Button("🎤", elem_classes=["bar-btn", "mic-pri"], min_width=90, interactive=MIC_ASR)
            with gr.Row(visible=False):
                next_btn = gr.Button("➡️", elem_classes=["bar-btn"], min_width=90, visible=False)
            with gr.Row(visible=False):
                start_btn = gr.Button("⭐", elem_classes=["bar-btn"], min_width=90)

            result = gr.HTML()
            feedback_audio = gr.Audio(
                type="filepath",
                autoplay=True,
                show_label=False,
            )
            question_audio = gr.Audio(
                type="filepath",
                autoplay=True,
                show_label=False,
            )

        silence_timer = gr.Timer(1, active=True)

        pick_reset = '<p class="child-pick" aria-label="number">?</p>'

        out_boot: list[gr.Component] = [
            image,
            session_state,
            question_audio,
            feedback_audio,
            result,
            num_picked,
            pick_show,
            start_btn,
            last_repeat_wav,
            awaiting,
            stim_t0,
            fired_10,
            did_20_once,
            num_pad_wrap,
        ]

        def _new_silence_timer() -> tuple[bool, float, bool, bool]:
            return True, time.time(), False, False

        def _pack_after_present(
            im,
            s2: TutorSession,
            play: str | None,
            rep: str | None,
            aw: bool,
            t0: float,
            f10: bool,
            d20: bool,
        ) -> list:
            return [
                im,
                s2,
                play,
                gr.update(value=None),
                _empty_result_html(),
                0,
                pick_reset,
                gr.update(),  # start_btn (hidden, session reset)
                rep,
                aw,
                t0,
                f10,
                d20,
                gr.update(elem_classes=["num-grid", "num-pad-outer"]),
            ]

        def on_bootstrap() -> list:
            aw, t0, f10, d20 = _new_silence_timer()
            sess = _load_session()
            im, _, s2, play, rep = do_present(sess, include_welcome=True)
            return _pack_after_present(im, s2, play, rep, aw, t0, f10, d20)

        out_next: list[gr.Component] = [
            image,
            result,
            question_audio,
            feedback_audio,
            session_state,
            num_picked,
            pick_show,
            start_btn,
            last_repeat_wav,
            awaiting,
            stim_t0,
            fired_10,
            did_20_once,
            num_pad_wrap,
        ]

        def on_next_fn(sess: TutorSession) -> list:
            aw, t0, f10, d20 = _new_silence_timer()
            im, res_h, play, s2, r1, _r2 = do_next(sess)
            return [
                im,
                res_h,
                play,
                gr.update(value=None),
                s2,
                0,
                pick_reset,
                gr.update(),
                r1,
                aw,
                t0,
                f10,
                d20,
                gr.update(elem_classes=["num-grid", "num-pad-outer"]),
            ]

        def _pack_after_scored(s2: TutorSession, msg: str, fb_wav: str | None) -> list:
            """Spoken feedback (child's language) then auto-advance: next image + TTS, new silence window."""
            u = gr.update()
            aw, t0, f10, d20 = _new_silence_timer()
            im, res_h, nplay, s3, r1, _r2 = do_next(s2)
            u_pad = gr.update(elem_classes=["num-grid", "num-pad-outer"])
            if im is None:
                return [
                    None,
                    res_h,
                    u,
                    gr.update(value=fb_wav, autoplay=True) if fb_wav else u,
                    s3,
                    0,
                    pick_reset,
                    u,
                    None,
                    False,
                    -1.0,
                    False,
                    False,
                    u_pad,
                ]
            comb = _concat_wav_files(fb_wav, nplay) if (fb_wav and nplay) else None
            if comb is not None:
                return [
                    im,
                    msg,
                    gr.update(value=comb, autoplay=True),
                    gr.update(value=None),
                    s3,
                    0,
                    pick_reset,
                    u,
                    r1,
                    aw,
                    t0,
                    f10,
                    d20,
                    u_pad,
                ]
            return [
                im,
                msg,
                gr.update(value=nplay, autoplay=True) if nplay else u,
                gr.update(value=fb_wav, autoplay=True) if fb_wav else u,
                s3,
                0,
                pick_reset,
                u,
                r1,
                aw,
                t0,
                f10,
                d20,
                u_pad,
            ]

        def on_check_tap(sess: TutorSession, n: int) -> list:
            msg, fb_wav, s2, _ok = do_score_tap(sess, n)
            return _pack_after_scored(s2, msg, fb_wav)

        def on_check_voice(sess: TutorSession, rec: object) -> list:
            msg, fb_wav, s2, ok = do_score_from_mic(sess, rec)  # type: ignore[arg-type]
            if ok is None:
                u = gr.update()
                return [u, u, u, u, s2, u, u, u, u, u, u, u, u, u]
            return _pack_after_scored(s2, msg, fb_wav)

        out_tick: list[gr.Component] = [
            question_audio,
            session_state,
            last_repeat_wav,
            image,
            result,
            feedback_audio,
            num_pad_wrap,
            awaiting,
            stim_t0,
            fired_10,
            did_20_once,
        ]

        def on_age(choice: str):
            return gr.update(
                elem_classes=["ux", "ux-tap" if choice == "56" else "ux-voice"]
            )

        def on_silence_tick(
            aw: bool,
            t0: float,
            f10: bool,
            d20: bool,
            last_rep: str | None,
            sess: TutorSession,
        ) -> list:
            n = len(out_tick)
            if t0 < 0.0 or not aw:
                return _n_noop(n)
            el = time.time() - t0
            if el < 10.0:
                return _n_noop(n)
            if el >= 20.0 and not d20:
                jumped = sess.jump_to_simpler_count()
                if not jumped:
                    sess.seek_item_id("g031_c")
                im, _, s2, play, rep = do_present(sess, include_welcome=False)
                if im is None or (play is None and rep is None):
                    return [
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        t0,
                        f10,
                        True,
                    ]
                aw2, t2, f10b, d20b = _new_silence_timer()
                return [
                    gr.update(value=play, autoplay=True) if play else gr.update(),
                    s2,
                    rep,
                    im,
                    _empty_result_html(),
                    gr.update(value=None),
                    gr.update(elem_classes=["num-grid", "num-pad-outer"]),
                    aw2,
                    t2,
                    f10b,
                    d20b,
                ]
            if el < 20.0 and not f10 and last_rep:
                return [
                    gr.update(value=last_rep, autoplay=True),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(elem_classes=["num-grid", "num-pad-outer", "pad-glow"]),
                    gr.update(),
                    gr.update(),
                    True,
                    d20,
                ]
            return _n_noop(n)

        age.change(on_age, [age], [main_ux])
        demo.load(
            on_bootstrap,
            None,
            out_boot,
        )
        start_btn.click(on_bootstrap, None, out_boot)
        check_tap.click(on_check_tap, [session_state, num_picked], out_next)
        check_mic.click(on_check_voice, [session_state, mic], out_next)
        next_btn.click(on_next_fn, [session_state], out_next)
        silence_timer.tick(
            on_silence_tick,
            [awaiting, stim_t0, fired_10, did_20_once, last_repeat_wav, session_state],
            out_tick,
        )

    return demo, theme, child_css


def _first_free_port(host: str, start: int, *, span: int = 32) -> int:
    """First bindable TCP port in [start, start+span) on *host*; or an OS-chosen free port if none."""
    h = "127.0.0.1" if not (host or "").strip() else host
    for port in range(start, start + span):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((h, port))
        except OSError:
            continue
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((h, 0))
        return int(s.getsockname()[1])


if __name__ == "__main__":
    print("Starting Math play (Gradio)…", flush=True)
    app, _theme, _css = build_ui()
    _host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    _want = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    _port = _first_free_port(_host, _want)
    if _port != _want:
        print(f"Port {_want} is in use; using {_port} instead.", flush=True)
    _kw: dict = {"server_name": _host, "server_port": _port, "theme": _theme, "css": _css}
    if _host in ("0.0.0.0", ""):
        open_url = f"http://127.0.0.1:{_port}"
    else:
        open_url = f"http://{_host}:{_port}"
    print(f"Open in your browser: {open_url}", flush=True)
    try:
        app.launch(**_kw)
    except TypeError:
        app.launch(server_name=_host, server_port=_port, theme=_theme, css=_css)
