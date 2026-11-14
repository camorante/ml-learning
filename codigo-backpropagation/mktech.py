"""Rellena tech.tpl.html con el código y las salidas reales → tech-body.html."""
import html, re, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t = open("../tech.tpl.html", encoding="utf-8").read()
def out(name):
    o = open(name.replace(".py", ".out"), encoding="utf-8").read().rstrip("\n")
    return "\n".join(l if l.startswith("#") else "# " + l for l in o.split("\n"))
t = re.sub(r"\{\{FILE:([\w.]+)\}\}", lambda m: html.escape(open(m[1], encoding="utf-8").read().rstrip("\n"), quote=False), t)
t = re.sub(r"\{\{OUT:([\w.]+)\}\}", lambda m: html.escape(out(m[1]), quote=False), t)
open("../tech-body.html", "w", encoding="utf-8").write(t)
