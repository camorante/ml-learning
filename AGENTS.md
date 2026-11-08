# AGENTS.md: «Aprender viendo», explicaciones visuales de ML

Guía para el agente que retome este proyecto en una conversación nueva. Explica qué hay, cómo está construido,
cómo se trabaja y qué falta. El autor es Carlos, ingeniero de software y datos en Zonar (telemetría GPS de flotas).
La serie se hizo para Zonar, así que el menú principal lleva su logo (`Zonar-Logo-2021-RGB.png`).

- **Repositorio:** GitHub `camorante/ml-learning`, rama `main`. Esta carpeta (`Gauss`) es el clon local.
- **Web:** GitHub Pages en `camorante.github.io/ml-learning`. `index.html` es la puerta de entrada.
- **Push:** Carlos hace `git push` él mismo. El agente solo deja los commits hechos.
- **Idioma:** se habla con Carlos en español. Todo el contenido existe en español (`-es`) y en inglés (`-en`).

---

## 1. Qué está hecho

Hay tres colecciones terminadas. Cada tema tiene 2 páginas por idioma (sencilla y técnica) y un vídeo con audio por idioma.

| Colección | Índice ES / EN | Temas | Estado |
|---|---|---|---|
| Las matemáticas debajo del ML | `matematicas-del-ml-es.html` / `math-under-ml-en.html` | 16 | Completa |
| Preparar los datos | `preparar-los-datos-es.html` / `preparing-the-data-en.html` | 8 | Completa |
| Serie ML visual | `serie-ml-visual-es.html` / `visual-ml-series-en.html` | 14 | Completa |

En `index.html` las colecciones aparecen en este orden: matemáticas, preparar los datos y serie ML. Esa página tiene un selector ES/EN; el resto de páginas van separadas por idioma.

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

Las redes neuronales se dejaron fuera de esta serie a propósito: tienen su propia colección (ver §5).

---

## 2. Archivos de cada tema (en la raíz del repositorio)

Para un tema con base `B` (en español) y `E` (en inglés):

| Archivo | Qué es |
|---|---|
| `B-es.html` / `E-en.html` | Página sencilla para alguien sin formación: dibujos interactivos y vídeo al final |
| `B-tecnico-es.html` / `E-technical-en.html` | Página técnica para programadores: ecuaciones (MathJax ya renderizado), widgets, código Python verificado y ejemplo resuelto |
| `B-con-audio-es.mp4` / `E-with-audio-en.mp4` | Vídeo de 60–90 s con música y efectos. Debe estar en la misma carpeta que las páginas |
| `B-manim-es.py`, `B-audio-es.py` / `E-manim-en.py`, `E-audio-en.py` | Escena Manim y banda sonora sintetizada (numpy/scipy) del vídeo |
| `codigo-<tema>/` | Scripts Python que generan cada bloque de código y su «salida real» de la técnica (solo en las colecciones de matemáticas y de preparar los datos) |

Otros archivos de la raíz:

- `index.html`: menú principal con el logo de Zonar y el selector ES/EN. Cada tarjeta lleva un contador del tipo «16 de 16 temas · 32 páginas · 16 vídeos», que hay que actualizar a mano en el HTML y también en el diccionario `T` del script.
- `README.md`: descripción corta del repositorio.
- `regresion-lineal-polinomial-y-logistica.pdf`: PDF de prueba. **No está en git** a propósito.
- `.gitignore`: ignora `_kit/`, `media/`, `__pycache__/` y `*.rar`.

### Cómo son las páginas

- Cada página es un único HTML autocontenido: CSS y JS en línea, el póster del vídeo en base64 y las ecuaciones ya convertidas a SVG.
- **Paleta común:** fondo `#F2F5FA`, tinta `#16202E`, azul «media» `#1D5FD1`, naranja «punto» `#DD6A26`, violeta «muestra» `#7563C9` y verde «bueno» `#1C8A5A`. Tiene modo oscuro por variables y `prefers-color-scheme`.
- **Fuentes:** Bricolage Grotesque para títulos, Atkinson Hyperlegible para el texto y JetBrains Mono para datos, todas de Google Fonts.
- **Eyebrow:** todas las páginas empiezan con «← Menú principal» (a `index.html#es` o `#en`), que añade `local.py`. Le siguen el enlace a la otra versión, el de «todos los temas» y el del otro idioma.
- **Página sencilla, estructura fija:**
  1. Hero.
  2. Juguete, con la caja «La situación del juguete», «Cómo leer el dibujo», tarjetas de decisión, «Prueba en este orden» y un mensaje dinámico.
  3. «Lo que acabas de ver».
  4. «Idea 1 de N»… (cada idea con su dibujo interactivo).
  5. «Todo junto».
  6. «En el mundo real»: 4 tarjetas con ejemplos de flota.
  7. «Conexiones».
  8. «Lo bueno / lo no tan bueno».
  9. «Mini diccionario» y el desplegable «Para curiosos».
  10. Vídeo «Para cerrar».
- **Página técnica:** secciones §1…§N con ecuaciones, al menos un widget, tablas, código, checklist y referencias reales. Termina con un **ejemplo resuelto** paso a paso en el que cada constante explica de dónde sale.

---

## 3. Cómo se construye: el kit `_kit/` (local, no está en git)

`_kit/` contiene todo el sistema de construcción. Lee primero **`_kit/LEEME-CONTINUAR.md`** y después las guías.

| Ruta | Qué es |
|---|---|
| `_kit/mt/GUIDE_AUTHOR.md` | Guía obligatoria para escribir un tema en español: estructura, tono, placeholders, build y vídeo |
| `_kit/mt/GUIDE_EN.md` + `_kit/en/GUIDE.md` | Guía de traducción al inglés: qué se traduce y qué no, formato de números, € → $ y vídeo |
| `_kit/mt/mK/`, `_kit/mt/mK-en/` (K=1..16) | Fuentes de cada tema de matemáticas: `simple-body.html`, `tech-body.html` (en los temas modernos se genera desde `tech.tpl.html` con `py/mktech.py` o `py/gen.py`), `ex.html` (ejemplo resuelto), `embed.json`/`tembed.json` (datos) y `py/` (scripts con su `.out`) |
| `_kit/vmK/` y `_kit/vmK/en/` | Escena Manim, audio, `marks.json` (tiempos de las marcas, que sincronizan la música) y `poster.jpg` |
| `_kit/d6/`, `_kit/vd6/` | Tema 6 de «Preparar los datos»: el modelo de referencia original del formato |
| `_kit/tree/treelib.js` | Librería JS de dibujo que se inserta en `/*TREELIB*/`: `el()`, `txt()`, `mulberry32()`, `gauss()` y `class Plot` |
| `_kit/log/simple-head.txt`, `_kit/build/tech-style.txt`, `_kit/ex/excss.html` | CSS base de la página sencilla, de la técnica y del ejemplo |
| `_kit/build/build.js`, `of3.js` | Prerrenderizan MathJax a SVG y detectan ecuaciones demasiado anchas (necesitan `npm i mathjax-full prismjs` en `build/`) |
| `_kit/reg/wrap.sh`, `_kit/en/wrap_en.sh` | Envuelven el cuerpo en el documento HTML completo (ES / EN) |
| `_kit/mt/build.sh` | `bash mt/build.sh mt/mK "Título" "Título, para programadores" [en]`: construye `full.html` y `w-tech.html` y saca capturas de escritorio y de móvil oscuro (400 px). Debe dar `errors []`, `phone scrollWidth 400` y ninguna línea de anchura |
| `_kit/mt/pub.py` | `python3 mt/pub.py es|en src:dst …`: rellena los placeholders (`M7_SIMPLE_URL`, `REG_TECH_URL`, `D4_SIMPLE_URL`, `MK_POSTER`…) con las URL de los artifacts publicados, para publicarlos en claude.ai |
| `_kit/mt/local.py` | Igual que `pub.py`, pero para las copias locales y de GitHub Pages: enlaces relativos con sufijo, nombres de vídeo finales y el enlace «← Menú principal». Tiene las listas `MES`/`MEN` de nombres base y el diccionario `VID` de vídeos |
| `_kit/mt/hubgen.py` | Plan de los 16 temas (lista `T`) y generador de los índices de la colección (`mt/hub-es.html`, `mt/hub-en.html`) a partir de `datos/index.html` y `en/hub.html` |
| `_kit/mt/urls.json`, `_kit/datos/urls.json`, `_kit/en/urls_en.json`, `_kit/hub/index.html` | IDs de los artifacts de claude.ai ya publicados (matemáticas, datos, inglés y serie ML) |

**Ojo:** las fuentes de la Serie ML visual y de «Preparar los datos» (salvo d6) **no están en el kit**. Solo están sus HTML finales en la raíz. Para retocarlas hay que editar el HTML final.

Los `_kit/_*.tgz` son restos de transferencias entre el ordenador de Carlos y el espacio de la nube. Se pueden borrar.

### Entorno (espacio de trabajo en la nube)

1. Pasa `_kit/` al scratchpad de la sesión: haz un `tar` en el dispositivo, `device_stage_files` y extráelo. Después cambia la ruta `BASE`/`B` en `mt/*.py`, `mt/build.sh`, `en/GUIDE.md` y `mt/GUIDE_*.md` (ahora apunta a un scratchpad antiguo).
2. Dependencias:
   - `cd build && npm i mathjax-full prismjs`.
   - Playwright y Chromium ya están en `/opt/pw-browsers/chromium`; usa `NODE_PATH=/opt/node-tools/node_modules`.
   - Para Manim: `apt-get install -y libpango1.0-dev pkg-config libcairo2-dev`, después `python3 -m venv /opt/mvenv --system-site-packages && /opt/mvenv/bin/pip install manim`.
   - La fuente Inter ya está instalada (la usan los vídeos).
3. Vídeo:

   ```
   cd vmK && flock /tmp/manim.lock /opt/mvenv/bin/manim -qh mK-manim.py Escena
   /opt/mvenv/bin/python mK-audio.py
   ffmpeg … -c:v copy -c:a aac -shortest NOMBRE.mp4
   ```

   El póster se saca con `ffmpeg -ss 2.5`. Usa siempre `flock`: la máquina tiene 2 núcleos y 7 GB.
4. Versiones instaladas: numpy 2.5, scipy 1.18 y scikit-learn 1.9. **No** están `statsmodels`, `torch` ni `jax`. Para redes neuronales habrá que comprobar si se puede instalar `torch` (CPU) o verificar con una implementación propia en numpy.
5. Las capturas no cargan Google Fonts (el proxy las bloquea). Es normal; la página publicada sí las carga.

### Flujo por pareja de temas (se hacen de 2 en 2)

1. **Escribir en español:** dos subagentes en paralelo, uno por tema, siguiendo `GUIDE_AUTHOR.md`. Cada uno revisa sus capturas con Read (escritorio y móvil oscuro, y widgets en varios estados con `el.js`) y fotogramas del vídeo.
2. **Revisión:** lee los informes y corrige lo que quede (etiquetas cortadas, cifras que contradicen la teoría…).
3. **Traducir:** dos subagentes con `GUIDE_EN.md`. Si encuentran un fallo de maquetación que también está en español, lo corrigen en los dos idiomas.
4. **Publicar en claude.ai** (Artifact). La primera pasada publica las 8 páginas con `pub.py`, con el `.mp4` en `files`. Después se apuntan los IDs en `urls.json`, se vuelve a ejecutar `pub.py` y se republica todo. Hace falta esa segunda pasada para que los enlaces cruzados queden rellenos.
5. **Índices:** en `hubgen.py`, pon `ready: True`, los nombres base y el título del ejemplo. Ejecuta `python3 mt/hubgen.py` y republica los dos índices.
6. **Copias locales:** añade los nombres base a `MES`/`MEN` y los vídeos a `VID` en `local.py`. Ejecuta `python3 mt/local.py es|en …` (que escribe en `mt/local/`) y copia los vídeos con su nombre final.
7. **Copiar a la carpeta Gauss:** haz un `tar` en la nube con las páginas, los vídeos, los `.py` del vídeo renombrados y `codigo-<tema>/`. Pásalo con `device_commit_files` y extráelo con `tar --overwrite`.
8. **Actualizar los contadores** de `index.html` (HTML y diccionario JS de los dos idiomas).
9. **Commit** en la carpeta Gauss (Carlos hace el push) y actualiza `_kit/` con el kit nuevo.

### Problemas conocidos de git en esta carpeta

El agente no tiene permiso para borrar en la carpeta conectada. Por eso git deja `.git/HEAD.lock` y `.git/objects/maintenance.lock` después de cada commit. Hay que renombrarlos (`mv .git/HEAD.lock .git/stale-HEAD-lock-N`) para que el siguiente comando de git funcione. Los `stale-*` y `tmp_obj_*` sobrantes se pueden borrar. Otra opción es pedir permiso de borrado con `device_request_delete_permission`.

---

## 4. Reglas del autor (obligatorias)

- **Página sencilla:** no supone ningún conocimiento previo. **Cada control interactivo explica qué significa y qué hacer con él.**
- **Ejemplos:** siempre de telemetría de flotas de camiones (consumo en L/100 km, ralentí, frenadas bruscas, paradas, averías, entregas…).
- **Dinero:** en **€** en español y en **$** en inglés, con las mismas cifras.
- **Números:** en español, coma decimal y espacio de miles; en inglés, punto decimal, coma de miles y sin espacio antes de %.
- **Página técnica:** termina con un **ejemplo resuelto realista, paso a paso**, que entienda alguien que no es matemático: cada constante, símbolo o factor de escala explica de dónde sale.
- **Código y salidas:** todo el código y todas sus salidas («# salida real») salen de ejecutarlo de verdad. Hay un módulo «desde cero» verificado contra numpy/scipy/sklearn, con el error máximo real.
- **Coherencia de números:** los de la sencilla, la técnica, el ejemplo y el vídeo deben cuadrar entre sí.
- **Simulaciones:** si un widget muestra una cifra calculada con pocas simulaciones al azar, usa suficientes para que no contradiga la teoría que explica la página.
- **Vídeo:** **solo con audio**, de 60 a 90 s (máximo 1:30), con la paleta oscura y la fuente Inter. Nada pegado al borde ni solapado.
- **Móvil:** a 400 px, sin desplazamiento horizontal ni etiquetas cortadas. Los arrastres en SVG no deben bloquear el desplazamiento de la página.
- **Inglés:** son páginas separadas, no un selector de idioma.
- **Logo:** el logo de Zonar va en `index.html`. Si se añade a otras páginas, en modo oscuro va sobre una placa blanca, porque las letras del logo son negras.

---

## 5. Lo que falta: la colección «Redes neuronales»

Es la cuarta colección, con el mismo formato y las mismas reglas: página sencilla, técnica, ejemplo resuelto, vídeo con audio, español e inglés, y de 2 en 2. Carlos dirá cuándo empezar.

| # | Tema | Contenido |
|---|---|---|
| 1 | La neurona | Suma ponderada más una activación (el perceptrón) |
| 2 | Funciones de activación | Sigmoide, tanh y ReLU, y por qué sin ellas todo sería lineal |
| 3 | La red multicapa | Cómo las capas ocultas «doblan el espacio» para separar lo que una recta no puede |
| 4 | Salida y pérdida | Softmax, entropía cruzada y error cuadrático |
| 5 | Backpropagation | La regla de la cadena recorriendo la red hacia atrás |
| 6 | Entrenar bien | Mini-batch, learning rate, momentum y Adam |
| 7 | Sobreajuste y regularización | Dropout, weight decay, early stopping, inicialización y batch norm |
| 8 | Redes convolucionales (CNN) | Filtros que recorren una imagen |
| 9 | Redes recurrentes y LSTM | Secuencias y series temporales; encaja muy bien con telemetría GPS |
| 10 | Embeddings | Categorías y palabras convertidas en vectores |
| 11 | Atención y Transformers | La pieza que hay detrás de los LLM |
| 12 | Autoencoders | Comprimir y reconstruir, y detectar anomalías por el error de reconstrucción |
| 13 | Proyecto completo | Una red de principio a fin sobre datos de flota, como el cierre de «Preparar los datos» |

### Cómo montarla (propuesta, a confirmar con Carlos)

- **Índice:** `redes-neuronales-es.html` / `neural-networks-en.html`. Añade una cuarta tarjeta en `index.html`, cambia «Tres colecciones» por «Cuatro» en los dos idiomas y pregunta a Carlos en qué posición va.
- **Kit nuevo:** `_kit/rn/` con `rK/`, `rK-en/` y `vrK/`, copiando `mt/build.sh` (es genérico), `GUIDE_AUTHOR.md` y `GUIDE_EN.md` adaptadas, y un `hubgen` propio con el plan de los 13 temas.
- **Placeholders:** usa un prefijo que no choque con los existentes, por ejemplo `R1_SIMPLE_URL` … `R13_TECH_URL` y `R1_POSTER`, y comprueba que la sustitución por longitud de `pub.py`/`local.py` no los confunda. Hay que ampliar `pub.py` y `local.py` (o hacer versiones propias) para que conozcan el prefijo nuevo y sigan rellenando `M*`, `D*` y la serie ML.
- **Enlaces que encajan:**
  - Tema 2 de matemáticas (producto punto) para la neurona.
  - Tema 3 (matrices) para las capas.
  - Tema 15 (entropía cruzada y softmax) para la salida y la pérdida.
  - Tema 16 (derivadas, regla de la cadena y el mini-autograd `Valor`) para backpropagation.
  - Tema 14 (máxima verosimilitud) para las pérdidas.
  - Temas 11 y 12 para la evaluación.
  - Serie ML: LOG (una neurona sigmoide es una regresión logística), GD (descenso de gradiente) y PCA (para los autoencoders).
  - Preparar los datos: D7 (escalado de variables) y D8 (proyecto completo, como modelo del tema 13).
- **Verificación del código:** si no se puede instalar `torch`, verifica las implementaciones numpy desde cero con diferencias finitas (como en el tema 16) y con `sklearn.neural_network.MLPClassifier`/`MLPRegressor` donde aplique.

---

## 6. Detalles pendientes menores (opcional)

- **Logo de Zonar:** solo está en `index.html`. Se puede llevar a los índices de colección o a todas las páginas.
- **Tema 15 de matemáticas, idea 2:** con el deslizador al 0 %, el punto naranja tapa un poco las etiquetas «0,00» y «0 %».
- **Tema 14 de matemáticas, idea 3:** las etiquetas del intervalo quedan encima de curvas grises. Se leen gracias al halo.
- **Tema 13 de matemáticas, juguete en móvil:** los puntos repetidos se apilan bastante altos.
- **Tema 10 de matemáticas:** en las figuras donde se arrastran puntos (ideas 1 y 4, y el widget del atípico de la técnica), en móvil la página no se desplaza si el dedo empieza encima de la figura.
