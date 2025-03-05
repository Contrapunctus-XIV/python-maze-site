from fasthtml.common import *
from dataclasses import dataclass
from utils.Maze import Maze, MazeDrawer, GenerationMethods
import traceback

flexbox = Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css"),
style = Style('.page-container { padding: 10px; } html { overflow-x: hidden; } .img-container { display: flex; justify-content: center; } #title-container { text-align: center; } #maze-form-container { padding-right: 20px; } #pre-gen-div { width: 500px; height: 500px; border: 1px solid; display: flex; align-items: center; justify-content: center; } #maze-img { border: 1px solid #0c438a; } .error-msg { color: red; } .hidden { display: none !important; } #solve-btn { background-color: red; border: 1px solid red; margin-bottom: var(--pico-spacing); width: 100%; }')
js = """
function loadMessage() {
    const btn = document.querySelector("#solve-btn");
    btn.classList.add("hidden");
    btn.textContent = "Résoudre";
    
    const div = document.querySelector('.img-container:not(.hidden)');
    div.innerHTML = '<div id="pre-gen-div"><p>Génération en cours...</p></div>';
}

function showSolveButton() {
    const btn = document.querySelector("#solve-btn");
    btn.classList.remove("hidden");
    btn.setAttribute("active", "false");
}

function showSolution() {
    const btn = document.querySelector("#solve-btn");
    
    const solDiv = document.querySelector("#sol-container");
    const srcDiv = document.querySelector("#src-container");
    const isActive = btn.getAttribute("active") === "true";
    
    srcDiv.classList.toggle("hidden", !isActive);
    solDiv.classList.toggle("hidden", isActive);
    btn.setAttribute("active", !isActive);
    btn.textContent = isActive ? "Résoudre" : "Cacher la résolution";
}
"""

app = FastHTML(hdrs=(picolink, flexbox, style))

MINIMAL_MAZE_SIZE = 3
MAXIMAL_MAZE_SIZE = 45

# structure du formulaire envoyé au backend pour la génération
@dataclass
class MazeForm:
    algorithm: str; size: int
    def __init__(self, algorithm: str = "", size: int = 0): store_attr()

@app.get("/{fname:path}.{ext:static}")
def static(fname: str, ext: str):
    return FileResponse(f'public/{fname}.{ext}')

@app.route("/")
def get():
    return RedirectResponse("/maze", status_code=302)

@app.route("/maze")
def get():
    return Title('Maze Generator'), Main(
        Div(H1('Générateur de labyrinthe', _id='title'), _id='title-container'),
        Div(
            Div(Div(
                Div(P("Vous n'avez encore rien généré..."), _id="pre-gen-div"), # contiendra les images
                cls='box img-container'), cls='col-xs-8', _id="all-img-container"),
            Div(Div(
                Form(
                    Label(
                        "Dimension",
                        Input(type='number', name='size', placeholder=f'Entre {MINIMAL_MAZE_SIZE} et {MAXIMAL_MAZE_SIZE}', _id='size', min=MINIMAL_MAZE_SIZE, max=MAXIMAL_MAZE_SIZE, required=True),
                    _for='size'),
                    Label(
                        "Algorithme",
                        Select(*[Option(m.value[1], value=m.value[0]) for m in GenerationMethods], name='algorithm', _id='algorithm', required=True),
                    _for='algorithm', name="algorithm"),
                    Button("Générer un labyrinthe", type="submit"),
                    Button("Résoudre", _id="solve-btn", type="button", cls="hidden", hx_on_click = "showSolution();", active="false"),
                hx_post="/maze/generate", hx_target="#all-img-container", hx_on_htmx_before_request="loadMessage();", hx_on_htmx_after_request="showSolveButton();"), cls='box', _id="maze-form-container"), cls='col-xs-4')
            , cls='row page-container')
        ), Script(js)
    
@app.route('/maze/generate')
def post(f: MazeForm):
    try:
        # vérification de la taille
        if f.size < MINIMAL_MAZE_SIZE or f.size > MAXIMAL_MAZE_SIZE:
            return Response('<p class="error-msg">Invalid maze size.</p>', status_code=400)
        
        method = None
        
        # parcours par valeur permettant de retrouver l'algorithme choisi dans GenerationMethods
        for algorithm in GenerationMethods:
            if f.algorithm == algorithm.value[0]:
                method = algorithm
                break
        
        # si une méthode inconnue a été fournie
        if method is None:
            return Response('<p class="error-msg">Unsupported algorithm.</p>', status_code=400)
        
        m = Maze(f.size, f.size)
        m.set_generation_method(method)
        m.generate()
        
        drawer = MazeDrawer(m)
        src, sol = drawer.draw()
        
        # renvoie un div contenant les deux canevas (sans solution et avec solution)
        return Div(Img(src=src, _id="maze-img"), cls="img-container", _id="src-container"), Div(Img(src=sol, _id="maze-img"), cls="img-container hidden", _id="sol-container")
    
    except Exception as e:
        return Response(f'<p class="error-msg">Internal Server Error: {traceback.format_exc()}</p>', status_code=500)
    
if __name__ == "__main__":
    serve()