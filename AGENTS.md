# AGENTS.md: «Aprender viendo», explicaciones visuales de ML

Guía para el agente que retome este proyecto en una conversación nueva. Explica qué hay, cómo está construido,
cómo se trabaja y qué falta. El autor es Carlos, ingeniero de software y datos en Zonar (telemetría GPS de flotas).
La serie se hizo para Zonar, así que el menú principal lleva su logo (`Zonar-Logo-2021-RGB.png`).

- **Repositorio:** GitHub `camorante/ml-learning`, rama `main`. El clon local es `D:\Programacion\IA\Gauss`.
  Es la única carpeta de trabajo; la copia antigua `z1-ai-kb` ya no existe.
- **Web:** GitHub Pages en `camorante.github.io/ml-learning`. `index.html` es la puerta de entrada.
- **Ramas y push:** el trabajo va en una rama (`feat/<nombre>`) con commits por unidad de trabajo. El merge a `main`
  y el `git push` se hacen **solo cuando Carlos lo pide**. Commits con Conventional Commits y sin atribución a IA.
- **Idioma:** se habla con Carlos en español. Todo el contenido existe en español (`-es`) y en inglés (`-en`).

---

## 1. Qué está hecho

Las cuatro colecciones están terminadas. Cada tema tiene 2 páginas por idioma (sencilla y técnica) y un vídeo con
audio por idioma.

| Colección | Índice ES / EN | Temas | Estado |
|---|---|---|---|
| Las matemáticas debajo del ML | `matematicas-del-ml-es.html` / `math-under-ml-en.html` | 16 | Completa |
| Preparar los datos | `preparar-los-datos-es.html` / `preparing-the-data-en.html` | 8 | Completa |
| Serie ML visual | `serie-ml-visual-es.html` / `visual-ml-series-en.html` | 14 | Completa |
| Redes neuronales | `redes-neuronales-es.html` / `neural-networks-en.html` | 13 | Completa |

En `index.html` las colecciones aparecen en este orden: matemáticas, preparar los datos, serie ML y redes
neuronales, en una cuadrícula de 2×2. Esa página tiene un selector ES/EN; el resto de páginas van separadas por idioma.

### Las matemáticas debajo del ML (prefijo de placeholders `M1…M16`)

| # | Español (base) | Inglés (base) |
|---|---|---|
| 1 | vectores-y-distancias | vectors-and-distances |
| 2 | producto-punto-y-coseno | dot-product-and-cosine |
| 3 | matrices-como-transformaciones | matrices-as-transformations |
| 4 | minimos-cuadrados-como-proyeccion | least-squares-as-a-projection |
| 5 | autovectores-y-autovalores | eigenvectors-and-eigenvalues |
| 6 | descomposicion-svd | the-svd |
| 7 | media-varianza-y-desviacion-tipica | mean-variance-and-standard-deviation |
| 8 | distribuciones | distributions |
| 9 | probabilidad-condicional-e-independencia | conditional-probability-and-independence |
| 10 | covarianza-y-correlacion | covariance-and-correlation |
| 11 | muestreo-y-teorema-central-del-limite | sampling-and-the-central-limit-theorem |
| 12 | intervalos-de-confianza-y-bootstrap | confidence-intervals-and-the-bootstrap |
| 13 | pruebas-de-hipotesis-y-p-valor | hypothesis-tests-and-p-values |
| 14 | maxima-verosimilitud | maximum-likelihood |
| 15 | entropia-y-entropia-cruzada | entropy-and-cross-entropy |
| 16 | derivadas-gradiente-y-regla-de-la-cadena | derivatives-gradient-and-the-chain-rule |

### Preparar los datos (prefijo `D1…D8`)

1. separar-antes-de-mirar / split-before-you-look
2. analisis-exploratorio / exploratory-data-analysis
3. duplicados-y-errores-de-registro / duplicates-and-logging-errors
4. valores-atipicos / outliers
5. datos-que-faltan / missing-data
6. clases-desbalanceadas / imbalanced-classes
7. preparar-variables / preparing-features
8. proyecto-completo / end-to-end-project

### Serie ML visual (placeholders con clave propia)

| Clave | Español | Inglés |
|---|---|---|
| REG | regresion-lineal-polinomial | linear-and-polynomial-regression |
| LOG | regresion-logistica | logistic-regression |
| GD | descenso-de-gradiente | gradient-descent |
| TREE | arboles-de-decision | decision-trees |
| ENS | random-forest-y-boosting | random-forest-and-boosting |
| XG | xgboost | xgboost |
| KNN | k-vecinos-mas-cercanos | k-nearest-neighbors |
| KM | k-means | k-means |
| PCA | pca | pca |
| SVM | maquinas-de-vectores-soporte | support-vector-machines |
| BY | bayes | bayes |
| GP | procesos-gaussianos | gaussian-processes |
| MK | cadenas-de-markov | markov-chains |
| KF | filtro-de-kalman | kalman-filter |

### Redes neuronales (prefijo `R1…R13`, índice `NN_SERIES_URL`)

| # | Español (base) | Inglés (base) |
|---|---|---|
| 1 | la-neurona | the-neuron |
| 2 | funciones-de-activacion | activation-functions |
| 3 | red-multicapa | the-multilayer-network |
| 4 | salida-y-perdida | output-and-loss |
| 5 | backpropagation | backpropagation |
| 6 | entrenar-bien | training-well |
| 7 | sobreajuste-y-regularizacion | overfitting-and-regularization |
| 8 | redes-convolucionales | convolutional-networks |
| 9 | redes-recurrentes-y-lstm | recurrent-networks-and-lstm |
| 10 | embeddings | embeddings |
| 11 | atencion-y-transformers | attention-and-transformers |
| 12 | autoencoders | autoencoders |
| 13 | proyecto-completo-con-redes | end-to-end-network-project |

El seguimiento de esta colección (tareas, evidencias y decisiones) está en `odd/tasks/neural-networks.md`.

---

## 2. Archivos de cada tema (en la raíz del repositorio)

Para un tema con base `B` (en español) y `E` (en inglés):

| Archivo | Qué es |
|---|---|
| `B-es.html` / `E-en.html` | Página sencilla para alguien sin formación: dibujos interactivos y vídeo al final |
| `B-tecnico-es.html` / `E-technical-en.html` | Página técnica para programadores: ecuaciones (MathJax ya renderizado), widgets, código Python verificado y ejemplo resuelto |
| `B-con-audio-es.mp4` / `E-with-audio-en.mp4` | Vídeo de 60–90 s con música y efectos. Debe estar en la misma carpeta que las páginas |
| `B-manim-es.py`, `B-audio-es.py` / `E-manim-en.py`, `E-audio-en.py` | Escena Manim y banda sonora sintetizada (numpy/scipy) del vídeo |
| `codigo-<tema>/` | Scripts Python que generan cada bloque de código y su «salida real» de la técnica. Solo `.py`, sin `.out` (colecciones de matemáticas, preparar los datos y redes neuronales) |

Otros archivos de la raíz:

- `index.html`: menú principal con el logo de Zonar y el selector ES/EN. Cada tarjeta lleva un contador del tipo
  «16 de 16 temas · 32 páginas · 16 vídeos», que hay que actualizar a mano en el HTML y también en el diccionario `T`
  del script (los dos idiomas).
- `odd/tasks/`: documentos de seguimiento de cada trabajo grande.
- `README.md`: descripción corta del repositorio.
- `.gitignore`: ignora `_kit/`, `media/`, `__pycache__/` y `*.rar`.

### Cómo son las páginas

- Cada página es un único HTML autocontenido: CSS y JS en línea, el póster del vídeo en base64 y las ecuaciones ya
  convertidas a SVG.
- **Paleta común:** fondo `#F2F5FA`, tinta `#16202E`, azul «media» `#1D5FD1`, naranja «punto» `#DD6A26`, violeta
  «muestra» `#7563C9` y verde «bueno» `#1C8A5A`. Tiene modo oscuro por variables y `prefers-color-scheme`.
- **Fuentes:** Bricolage Grotesque para títulos, Atkinson Hyperlegible para el texto y JetBrains Mono para datos,
  todas de Google Fonts.
- **Eyebrow:** todas las páginas empiezan con «← Menú principal» (a `index.html#es` o `#en`), que añade `local.py`.
  Le siguen el enlace a la otra versión, el de «todos los temas» y el del otro idioma.
- **Página sencilla, estructura fija:**
  1. Hero.
  2. Juguete, con la caja «La situación del juguete», «Cómo leer el dibujo», tarjetas de decisión, «Prueba en este
     orden» y un mensaje dinámico.
  3. «Lo que acabas de ver».
  4. «Idea 1 de N»… (cada idea con su dibujo interactivo).
  5. «Todo junto».
  6. «En el mundo real»: 4 tarjetas con ejemplos de flota.
  7. «Conexiones».
  8. «Lo bueno / lo no tan bueno».
  9. «Mini diccionario» y el desplegable «Para curiosos».
  10. Vídeo «Para cerrar»: «Todo en un minuto» si dura menos de 1:15, «Todo en minuto y medio» si dura más.
- **Página técnica:** secciones §1…§N con ecuaciones, al menos un widget, tablas, código, checklist y referencias
  reales. Termina con un **ejemplo resuelto** paso a paso en el que cada constante explica de dónde sale.

---

## 3. Cómo se construye: el kit `_kit/` (local, no está en git)

Todo se construye **en local, en Windows**, con Git Bash. Ya no se usa el espacio de trabajo en la nube.

### Entorno

| Pieza | Dónde |
|---|---|
| Python 3.12 (numpy, scipy, scikit-learn, manim, torch CPU) | venv `D:\Trabajo\Zonar\z1-venv` → `PY=/d/Trabajo/Zonar/z1-venv/Scripts/python.exe` |
| ffmpeg 9 | `ffmpeg` en el PATH, o la ruta completa en `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_…\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe` |
| Node: mathjax-full, prismjs, playwright (con Chromium) | `_kit/build/node_modules` → `NODE_PATH=/d/Programacion/IA/Gauss/_kit/build/node_modules` |
| Fuente Inter (vídeos) | instalada en Windows |

Reglas del entorno:

- Exporta siempre `PYTHONUTF8=1` y abre los archivos con `encoding='utf-8'` (Windows usa cp1252 por defecto).
- No hay `python3`, `flock` ni LaTeX. En Manim solo se usa `Text` (nada de `MathTex`).
- Los parches con TeX, `\n` o comillas se escriben en un archivo con cadenas raw: los heredocs de Git Bash rompen
  `\\` y el texto no ASCII. No uses `sd` en textos con `$` (lo toma como grupo de captura).
- La máquina tiene 16 núcleos: se pueden renderizar varios vídeos a la vez.

### Qué hay en el kit

| Ruta | Qué es |
|---|---|
| `_kit/rn/GUIDE_AUTHOR.md` | Guía para escribir un tema nuevo en español, con el entorno local y las lecciones aprendidas. **Es la guía de referencia actual** |
| `_kit/rn/GUIDE_EN.md` + `_kit/en/GUIDE.md` | Guía de traducción al inglés (la primera manda en rutas y entorno) |
| `_kit/rn/rK/`, `_kit/rn/rK-en/` (K=1..13) | Fuentes de cada tema de redes: `simple.tpl.html` + `simple.js` → `simple-body.html` (con `py/mksimple.py`), `tech.tpl.html` → `tech-body.html` (con `py/mktech.py`), `ex.html`, `embed.json`/`tembed.json`, `py/` (scripts con su `.out`) y `rebuild.sh` |
| `_kit/vrK/`, `_kit/vrK/en/` | Vídeo de cada tema de redes: escena Manim, audio, `marks.json` y `poster.jpg` |
| `_kit/rn/hubgen.py` | Plan de los 13 temas y generador de los índices de la colección (`rn/hub/`) |
| `_kit/rn/sync_root.sh` | Regenera índices y copias locales y copia a la raíz páginas, vídeos, scripts del vídeo y `codigo-<base>/` |
| `_kit/mt/mK/`, `_kit/mt/mK-en/`, `_kit/vmK/` | Fuentes de los 16 temas de matemáticas (modelos de referencia) |
| `_kit/mt/build.sh` | `bash mt/build.sh <dir> "Título" "Título, para programadores" [en]`: construye `full.html` y `w-tech.html` y saca capturas de escritorio y de móvil oscuro (400 px). Sirve para `mt/mK` y `rn/rK` |
| `_kit/mt/local.py` | Rellena los placeholders para las copias de la raíz (enlaces relativos, vídeos, «← Menú principal»). Conoce `M*`, `D*`, la serie ML y `R*` (`local.py es|en rn`) |
| `_kit/mt/hubgen.py` | Índices de la colección de matemáticas |
| `_kit/mt/pub.py` | Rellena placeholders para publicar en claude.ai (Artifact). Ya no forma parte del flujo; solo si Carlos lo pide |
| `_kit/d6/`, `_kit/vd6/` | Tema 6 de «Preparar los datos»: el modelo original del formato. `d6/shot.js` y `d6/el.js` hacen las capturas |
| `_kit/rn/r3/measure.js` | Comprueba a 400 px que ninguna ecuación de MathJax se sale de su caja (`of3.js` no lo detecta todo) |
| `_kit/tree/treelib.js` | Librería JS de dibujo que se inserta en `/*TREELIB*/`: `el()`, `txt()`, `mulberry32()`, `gauss()` y `class Plot` |
| `_kit/log/simple-head.txt`, `_kit/build/tech-style.txt`, `_kit/ex/excss.html` | CSS base de la sencilla, de la técnica y del ejemplo |

Los scripts de `mt/` calculan su carpeta base a partir de su propia ubicación, así que funcionan desde cualquier sitio.

**Ojo:** las fuentes de la Serie ML visual y de «Preparar los datos» (salvo d6) **no están en el kit**. Solo están sus
HTML finales en la raíz. Para retocarlas hay que editar el HTML final.

### Flujo de trabajo de un tema (así se hizo la colección de redes)

1. **Escribir en español:** un subagente por tema (pueden ir en paralelo, en carpetas distintas), siguiendo
   `_kit/rn/GUIDE_AUTHOR.md`. Cada uno revisa capturas (escritorio y móvil oscuro, widgets en varios estados con
   `el.js`) y fotogramas del vídeo.
2. **Revisión del agente principal:** volver a construir (`errors []`, `phone scrollWidth 400`), mirar 1–2
   fotogramas y rehacer a mano las cifras clave del informe.
3. **Traducir:** un subagente por tema con `GUIDE_EN.md`. Traduce las entradas del generador, vuelve a ejecutar los
   scripts en `rK-en/py` (los números deben salir idénticos) y arregla en los dos idiomas los fallos de maquetación.
4. **Índices y contadores:** en `_kit/rn/hubgen.py` pon `True` en el tema; actualiza los contadores de
   `index.html` (HTML y diccionario `T`, ES y EN).
5. **Sincronizar:** `bash _kit/rn/sync_root.sh`. Después restaura con `git checkout` los archivos de la raíz que
   solo cambian en finales de línea (`git diff --ignore-cr-at-eol --quiet -- <archivo>`).
6. **Comprobar:** enlaces locales rotos, placeholders sin rellenar (`[A-Z0-9]+_(SIMPLE|TECH|POSTER)…`), móvil a
   400 px sin errores de JS.
7. **Commit** en la rama en dos unidades: contenido de los temas, e índices + contadores + documento `odd/`.
   Merge a `main` y push solo si Carlos lo pide.

### Notas de git

- `core.autocrlf=true`: los avisos «LF will be replaced by CRLF» son normales.
- `gentle-ai review` no funciona en esta carpeta porque `.git` pertenece a otro usuario de Windows. No cambies los
  permisos: es decisión de Carlos. En la práctica estas páginas superan el límite de contexto del revisor.

---

## 4. Reglas del autor (obligatorias)

- **Página sencilla:** no supone ningún conocimiento previo. **Cada control interactivo explica qué significa y qué
  hacer con él.**
- **Ejemplos:** siempre de telemetría de flotas de camiones (consumo en L/100 km, ralentí, frenadas bruscas, paradas,
  averías, entregas…).
- **Dinero:** en **€** en español y en **$** en inglés, con las mismas cifras.
- **Números:** en español, coma decimal y espacio de miles; en inglés, punto decimal, coma de miles y sin espacio
  antes de %.
- **Página técnica:** termina con un **ejemplo resuelto realista, paso a paso**, que entienda alguien que no es
  matemático: cada constante, símbolo o factor de escala explica de dónde sale.
- **Código y salidas:** todo el código y todas sus salidas («# salida real») salen de ejecutarlo de verdad, con
  semillas fijas. Hay un módulo «desde cero» verificado contra numpy/scipy/sklearn/torch, con el error máximo real.
- **Coherencia de números:** los de la sencilla, la técnica, el ejemplo y el vídeo deben cuadrar entre sí. Los
  formateadores de porcentaje en JS redondean primero la fracción (`f1(100*+p.toFixed(d+2),d)`) para coincidir con
  Python.
- **Honestidad:** si un resultado real contradice lo esperado (la red no gana al modelo base, el mapa no separa las
  anomalías…), se cuenta tal cual.
- **Simulaciones:** si un widget muestra una cifra calculada con pocas simulaciones al azar, usa suficientes para que
  no contradiga la teoría que explica la página.
- **Vídeo:** **solo con audio**, de 60 a 90 s (máximo 1:30), con la paleta oscura y la fuente Inter. Nada pegado al
  borde ni solapado; el título cambia en la misma animación que el dibujo; etiquetas que se pisan se separan.
- **Móvil:** a 400 px, sin desplazamiento horizontal ni etiquetas cortadas. Los arrastres en SVG no deben bloquear el
  desplazamiento de la página.
- **Inglés:** son páginas separadas, no un selector de idioma.
- **Referencias:** obras conocidas, con revista o congreso y año exactos (NIPS antes de 2018, NeurIPS desde 2018).
- **Logo:** el logo de Zonar va en `index.html`. Si se añade a otras páginas, en modo oscuro va sobre una placa
  blanca, porque las letras del logo son negras.

---

## 5. Pendiente

### Verificar las bibliografías (aplazado por Carlos hasta terminar la colección)

Las referencias de las páginas técnicas de **todas** las colecciones se escribieron de memoria, sin consultarlas.
Hay que comprobarlas de una vez contra fuentes primarias (DOI en Crossref, la editorial, arXiv) y corregirlas en
ES y EN. Detalle ya conocido: Krizhevsky 2012 aparece como NeurIPS; ese año el congreso se llamaba NIPS.

### Detalles menores (opcional)

- **Logo de Zonar:** solo está en `index.html`. Se puede llevar a los índices de colección o a todas las páginas.
- **Tema 15 de matemáticas, idea 2:** con el deslizador al 0 %, el punto naranja tapa un poco las etiquetas «0,00» y
  «0 %».
- **Tema 14 de matemáticas, idea 3:** las etiquetas del intervalo quedan encima de curvas grises. Se leen gracias al
  halo.
- **Tema 13 de matemáticas, juguete en móvil:** los puntos repetidos se apilan bastante altos.
- **Tema 10 de matemáticas:** en las figuras donde se arrastran puntos (ideas 1 y 4, y el widget del atípico de la
  técnica), en móvil la página no se desplaza si el dedo empieza encima de la figura.
- **Redes, tema 12:** en el mapa del código, tocar uno de los puntos normales sin ventana guardada muestra la curva del
  decodificador sin la entrada negra (así se generó `toys.py`).
- **Redes, tema 13, idea 6:** la etiqueta «95 % de las réplicas» queda sobre las barras en móvil; se lee por el halo.
- `_kit/LEEME-CONTINUAR.md` y `_kit/mt/GUIDE_AUTHOR.md` siguen describiendo el flujo antiguo en la nube; la guía
  vigente es `_kit/rn/GUIDE_AUTHOR.md`.
