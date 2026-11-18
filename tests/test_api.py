from fastapi.testclient import TestClient

from src.api import MAX_K, MAX_PREGUNTA, crear_app
from src.troceado import Trozo

CONFIG = {"k": 3, "modo_busqueda": "hibrido"}


class IndiceFalso:
    def __init__(self):
        self.trozos = [Trozo(0, "a.pdf", 1, "texto a1"), Trozo(1, "a.pdf", 2, "texto a2"),
                       Trozo(2, "b.pdf", 1, "texto b1")]
        self.llamadas = []

    def buscar(self, pregunta, k, modo):
        self.llamadas.append((pregunta, k, modo))
        return [(t, 0.5) for t in self.trozos[:k]]


class AgenteFalso:
    def __init__(self, resultado=None):
        self.resultado = resultado or {
            "estado": "respondida", "respuesta": "30 días.", "cita_textual": "En los 30 primeros días",
            "fuentes": [{"fuente": "a.pdf", "pagina": 1, "trozo_id": 0}], "recuperados": [0, 1]}

    def invoke(self, entrada):
        return {"resultado": self.resultado}


def _cliente(indice=None, agente=None):
    return TestClient(crear_app(indice or IndiceFalso(), CONFIG, agente))


def test_salud():
    r = _cliente(agente=AgenteFalso()).get("/salud")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "documentos": 2, "fragmentos": 3, "agente_disponible": True}
    assert _cliente().get("/salud").json()["agente_disponible"] is False


def test_documentos():
    assert _cliente().get("/documentos").json() == [{"fuente": "a.pdf", "fragmentos": 2},
                                                   {"fuente": "b.pdf", "fragmentos": 1}]


def test_buscar_devuelve_fragmentos_y_usa_el_modo_del_config():
    indice = IndiceFalso()
    r = _cliente(indice).post("/buscar", json={"pregunta": "  hola  ", "k": 2})
    assert r.status_code == 200
    assert [f["fuente"] for f in r.json()] == ["a.pdf", "a.pdf"]
    assert r.json()[0] == {"fuente": "a.pdf", "pagina": 1, "puntuacion": 0.5, "texto": "texto a1"}
    assert indice.llamadas == [("hola", 2, "hibrido")]


def test_k_por_defecto_es_5():
    indice = IndiceFalso()
    _cliente(indice).post("/buscar", json={"pregunta": "x"})
    assert indice.llamadas[0][1] == 5


def test_entradas_invalidas_dan_422_sin_buscar():
    indice = IndiceFalso()
    cliente = _cliente(indice)
    for cuerpo in ({"pregunta": ""}, {"pregunta": "   "}, {"pregunta": "x" * (MAX_PREGUNTA + 1)},
                   {"pregunta": "x", "k": 0}, {"pregunta": "x", "k": MAX_K + 1}, {}):
        assert cliente.post("/buscar", json=cuerpo).status_code == 422, cuerpo
    assert indice.llamadas == []


def test_preguntar_devuelve_respuesta_con_cita_y_fuentes():
    r = _cliente(agente=AgenteFalso()).post("/preguntar", json={"pregunta": "¿Plazo?"})
    assert r.status_code == 200
    datos = r.json()
    assert datos["estado"] == "respondida" and datos["cita_textual"] == "En los 30 primeros días"
    assert datos["fuentes"] == [{"fuente": "a.pdf", "pagina": 1, "trozo_id": 0}]


def test_preguntar_rechazada():
    rechazo = {"estado": "rechazada", "motivo": "sin_informacion",
               "respuesta": "No he encontrado esa información en los documentos.",
               "fuentes": [], "recuperados": []}
    datos = _cliente(agente=AgenteFalso(rechazo)).post("/preguntar", json={"pregunta": "¿?"}).json()
    assert datos["estado"] == "rechazada" and datos["motivo"] == "sin_informacion"
    assert datos["cita_textual"] is None


def test_preguntar_sin_agente_da_503():
    r = _cliente().post("/preguntar", json={"pregunta": "¿Plazo?"})
    assert r.status_code == 503
    assert "GEMINI_API_KEY" in r.json()["detail"]


def test_la_documentacion_se_genera_sola():
    esquema = _cliente().get("/openapi.json").json()
    assert {"/salud", "/documentos", "/buscar", "/preguntar"} <= set(esquema["paths"])
