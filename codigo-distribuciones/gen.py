"""Genera tech-body.html desde tech.tpl.html: inserta el código y la salida REAL de cada script."""
import re, subprocess, html, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t = open("../tech.tpl.html").read()
def rep(m):
    kind, f = m.group(1), m.group(2)
    if kind == "FILE":
        return html.escape(open(f).read().rstrip("\n"), quote=False)
    out = subprocess.run(["python3", f], capture_output=True, text=True, check=True).stdout.rstrip("\n")
    return html.escape("\n".join("# " + l for l in out.split("\n")), quote=False)
t = re.sub(r"\{\{(FILE|OUT):([\w.]+)\}\}", rep, t)
open("../tech-body.html", "w").write(t)
print("ok", len(t))
