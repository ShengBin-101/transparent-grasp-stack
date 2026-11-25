"""Tests the four tmux session files: adapted.yaml, baseline.yaml and raw.yaml are remake_grasp.yaml with
the preset fixed, and scripts/rg_session.sh leaves one pipeline session. Static checks plus the session
script against a fake tmux -- nothing is started and no real process is signalled.

    python3 tests/test_tmuxp.py          # host or container
"""
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


def panes(conf, window):
    w = next(w for w in conf["windows"] if w["window_name"] == window)
    return [p if isinstance(p, str) else p.get("shell_command") for p in w["panes"]]


generic = yaml.safe_load((ROOT / "tmuxp/remake_grasp.yaml").read_text())
CLEAN = "env -u MASK -u REMAKE -u TARGETS "
# session file -> (pipeline:=, launch arguments added to the remake and pipeline panes, the run pane's banner)
PRESETS = {"adapted": ("adapted", "", "ADAPTED"), "baseline": ("baseline", "", "BASELINE"), "raw": ("raw", "", "RAW")}
for name, (pipe, extra, banner) in PRESETS.items():
    conf = yaml.safe_load((ROOT / f"tmuxp/{name}.yaml").read_text())
    sess = f"remake_grasp_{name}"
    ck(f"{name}.yaml: session {sess}, its before_script keeps it and closes the others",
       conf["session_name"] == sess and conf["before_script"] == f"/opt/remake_grasp/scripts/rg_session.sh {sess}"
       and conf["start_directory"] == generic["start_directory"], (conf["session_name"], conf["before_script"]))
    # the generic file's node panes with the preset fixed, SBG, shell overrides dropped
    want = []
    for p in panes(generic, "nodes"):
        p = p.replace("${PIPELINE:-adapted}", pipe).replace("${DETECTOR:-sbg}", "sbg")
        if "remake.launch.py" in p or "pipeline.launch.py" in p:
            p += extra
        if "remake.launch.py" in p or "pipeline.launch.py" in p or "sbg.launch.py" in p:
            p = CLEAN + p
        want.append(p)
    got = panes(conf, "nodes")
    ck(f"{name}.yaml: the same node panes as remake_grasp.yaml, with pipeline:={pipe}{extra}", got == want,
       [(g, w) for g, w in zip(got, want) if g != w] or (len(got), len(want)))
    ck(f"{name}.yaml: no ${{PIPELINE}} / ${{MASK}} / ${{REMAKE}} / ${{TARGETS}} / ${{DETECTOR}} left in the node panes",
       not re.search(r"\$\{(PIPELINE|MASK|REMAKE|TARGETS|DETECTOR)", " ".join(got)), got)
    ck(f"{name}.yaml: the same windows, show_run pane and NUC pane",
       [w["window_name"] for w in conf["windows"]] == [w["window_name"] for w in generic["windows"]]
       and panes(conf, "run")[1:] == panes(generic, "run")[1:] and panes(conf, "nuc") == panes(generic, "nuc"))
    ck(f"{name}.yaml: the run pane says which pipeline this is",
       any(f'echo "{banner} ' in line for line in panes(conf, "run")[0]), panes(conf, "run")[0])
ck("remake_grasp.yaml: before_script closes the other sessions too",
   generic["before_script"] == "/opt/remake_grasp/scripts/rg_session.sh remake_grasp", generic["before_script"])
try:
    from tmuxp import config as tconf
    for name in ("remake_grasp", *PRESETS):
        c = tconf.expand(yaml.safe_load((ROOT / f"tmuxp/{name}.yaml").read_text()))
        ck(f"tmuxp accepts {name}.yaml", tconf.validate_schema(c) is True)
except ImportError:
    print("  (tmuxp not installed here: schema check skipped; run inside the container)")

# --- rg_session.sh with a fake tmux and a fake stop_nodes.sh ---------------------------------------
t = Path(tempfile.mkdtemp())
(t / "scripts").mkdir()
(t / "bin").mkdir()
(t / "scripts/rg_session.sh").write_text((ROOT / "scripts/rg_session.sh").read_text())
log = t / "log"
(t / "scripts/stop_nodes.sh").write_text(f"#!/bin/bash\necho stop_nodes >> {log}\n")
(t / "bin/tmux").write_text(f"""#!/bin/bash
if [ "$1" = list-sessions ]; then printf 'remake_grasp\\nremake_grasp_baseline\\nremake_grasp_adapted\\nother\\n'; fi
if [ "$1" = kill-session ]; then echo "kill $3" >> {log}; fi
""")
for f in ("scripts/rg_session.sh", "scripts/stop_nodes.sh", "bin/tmux"):
    (t / f).chmod((t / f).stat().st_mode | stat.S_IEXEC)
env = dict(os.environ, PATH=f"{t / 'bin'}:{os.environ['PATH']}")
for keep, want in (("remake_grasp_baseline", ["stop_nodes", "kill remake_grasp", "kill remake_grasp_adapted"]),
                   ("", ["stop_nodes", "kill remake_grasp", "kill remake_grasp_baseline", "kill remake_grasp_adapted"])):
    log.write_text("")
    r = subprocess.run(["bash", str(t / "scripts/rg_session.sh")] + ([keep] if keep else []), env=env)
    got = log.read_text().split("\n")[:-1]
    ck(f"rg_session.sh {keep or '(no argument: rg-down)'}: nodes stopped first, then "
       f"{'the other' if keep else 'every'} remake_grasp* session closed, 'other' left alone",
       r.returncode == 0 and got == want, got)

alias = (ROOT / "docker/env.sh").read_text()
ck("env.sh: an rg-<name> alias per session file and rg-down (rg_session.sh)",
   all(f"alias rg-{name.replace('_', '-')}='tmuxp load -y /opt/remake_grasp/tmuxp/{name}.yaml'" in alias for name in PRESETS)
   and "alias rg-down='/opt/remake_grasp/scripts/rg_session.sh'" in alias)
ck("tmuxp/: no session file the test does not know", sorted(f.stem for f in (ROOT / "tmuxp").glob("*.yaml"))
   == sorted(["remake_grasp", *PRESETS]), sorted(f.stem for f in (ROOT / "tmuxp").glob("*.yaml")))

print("\nFAILED: " + ", ".join(fail) if fail else "\nall passed")
sys.exit(1 if fail else 0)
