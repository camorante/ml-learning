"""simple.tpl.html + simple.js → simple-body.html (el título del vídeo según su duración, de marks.json)."""
import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t = open("../simple.tpl.html", encoding="utf-8").read()
mk = "../../../vr5/marks.json"
dur = json.load(open(mk, encoding="utf-8"))["total"][0] if os.path.exists(mk) else 70
t = t.replace("VIDEO_HEADING", "Todo en un minuto" if dur < 75 else "Todo en minuto y medio")
open("../simple-body.html", "w", encoding="utf-8").write(t.replace("/*SCRIPT*/", open("../simple.js", encoding="utf-8").read()))
print("vídeo", dur, "s")
