"""simple.tpl.html + simple.js → simple-body.html"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t = open("../simple.tpl.html", encoding="utf-8").read()
open("../simple-body.html", "w", encoding="utf-8").write(t.replace("/*SCRIPT*/", open("../simple.js", encoding="utf-8").read()))
