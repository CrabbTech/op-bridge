"""Command line for humans: run the server, probe the device, set the mode and constraints."""
from __future__ import annotations

import argparse
import json
import secrets
import sys

from .session import Config, Constraints, HOME, MODES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="op-bridge", description="MCP bridge to a Teenage Engineering OP-1 field")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--http", action="store_true", help="Streamable HTTP instead of stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--path-secret", default=None, help="random path segment so the URL itself is the secret; 'auto' generates one")
    s.add_argument("--public", action="store_true", help="accept any Host header (needed behind a tunnel such as cloudflared or ngrok)")

    st = sub.add_parser("status", help="show config and device presence")
    m = sub.add_parser("mode", help="set freeform or guided mode"); m.add_argument("mode", choices=MODES)
    c = sub.add_parser("constraints", help="set or clear constraints (human side; the model can only tighten)")
    c.add_argument("action", choices=["set", "clear", "show"]); c.add_argument("pairs", nargs="*", help="key=value, e.g. key='D minor' tempo=96 allow_master=false allowed_slots='synth 1,synth 2'")
    ch = sub.add_parser("channel", help="tell the bridge which MIDI channel the Field is set to"); ch.add_argument("channel", type=int)
    jd = sub.add_parser("judge", help="ranking judge for sound search: auto (Jev when a key is set), local (never call out), jev (require Jev)"); jd.add_argument("mode", choices=["auto", "local", "jev"])
    p = sub.add_parser("probe", help="characterize the connected Field"); p.add_argument("args", nargs="*")
    sub.add_parser("sessions", help="list sessions")
    u = sub.add_parser("use", help="switch session"); u.add_argument("name")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .server import run
        secret = secrets.token_urlsafe(18) if a.path_secret == "auto" else a.path_secret
        if a.http:
            url = f"http://{a.host}:{a.port}/{secret + '/' if secret else ''}mcp"
            print(f"op-bridge MCP server on {url}", file=sys.stderr)
        run("streamable-http" if a.http else "stdio", a.host, a.port, secret, a.public)
        return 0
    cfg = Config.load()
    if a.cmd == "status":
        from .device import find_midi_output, find_audio_input
        from .judge import active_judge
        print(json.dumps({"home": HOME, "mode": cfg.mode, "judge": cfg.judge, "active_judge": active_judge(), "midi_channel": cfg.midi_channel, "session": cfg.current_session, "constraints": cfg.constraints.__dict__,
                          "field_midi": find_midi_output(), "field_audio": find_audio_input()}, indent=1))
        return 0
    if a.cmd == "mode":
        cfg.mode = a.mode; cfg.save(); print(f"mode: {a.mode}"); return 0
    if a.cmd == "channel":
        cfg.midi_channel = a.channel; cfg.save(); print(f"midi channel: {a.channel}"); return 0
    if a.cmd == "judge":
        cfg.judge = a.mode; cfg.save(); print(f"judge mode: {a.mode}"); return 0
    if a.cmd == "constraints":
        if a.action == "clear":
            cfg.constraints = Constraints(); cfg.save(); print("constraints cleared"); return 0
        if a.action == "set":
            for pair in a.pairs:
                k, _, v = pair.partition("=")
                if k not in Constraints.__dataclass_fields__:
                    print(f"unknown constraint {k}", file=sys.stderr); return 2
                cur = getattr(cfg.constraints, k)
                if isinstance(cur, bool) or k.startswith("allow_"):
                    val: object = v.lower() in ("1", "true", "yes", "on")
                elif k in ("allowed_slots", "allowed_cc_params"):
                    val = [x.strip() for x in v.split(",") if x.strip()]
                elif k == "tempo_range":
                    val = [float(x) for x in v.split(",")]
                elif k == "extra_pitch_classes":
                    val = [int(x) for x in v.split(",")]
                elif k in ("tempo", "max_take_seconds"):
                    val = float(v)
                elif k == "max_polyphony":
                    val = int(v)
                else:
                    val = v
                setattr(cfg.constraints, k, val)
            cfg.save()
        print(json.dumps(cfg.constraints.__dict__, indent=1)); return 0
    if a.cmd == "probe":
        from .probe import main as probe_main
        return probe_main(a.args)
    if a.cmd == "sessions":
        from .session import list_sessions
        print("\n".join(list_sessions()) or "(none)"); return 0
    if a.cmd == "use":
        cfg.current_session = a.name; cfg.save(); print(f"session: {a.name}"); return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
