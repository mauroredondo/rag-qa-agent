from src.agente import MAX_INTENTOS, MENSAJE_RECHAZO, cita_es_literal, construir_agente, construir_prompt, preguntar
from src.llm import Respuesta
from src.troceado import Trozo

TEXTO = ("Derecho de desistimiento En los 30 primeros días desde la contra- tación del seguro "
         "el Tomador podrá desistir sin necesidad de que indique los motivos.")
CITA_BUENA = "En los 30 primeros días desde la contratación del seguro el Tomador podrá desistir"
CONFIG = {"k": 2, "modo_busqueda": "denso", "dominio": "seguros", "descripcion": "pólizas de seguro"}


class IndiceFalso:
    def buscar(self, pregunta, k, modo):
        return [(Trozo(7, "ing.pdf", 51, TEXTO), 0.9), (Trozo(8, "ley.pdf", 3, "otro texto sin relacion alguna"), 0.5)]


class LlmFalso:
    """Devuelve las respuestas que se le den, en orden, y guarda los prompts."""
    def __init__(self, *respuestas):
        self.respuestas, self.prompts = list(respuestas), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.respuestas.pop(0)


def _ok(cita=CITA_BUENA, citas=(1,)):
    return Respuesta(responde=True, respuesta="30 días.", citas=list(citas), cita_textual=cita)


def _agente(llm):
    return construir_agente(IndiceFalso(), llm, CONFIG)


def test_responde_cuando_la_cita_es_literal_aunque_el_pdf_tenga_guiones():
    r = preguntar(_agente(LlmFalso(_ok())), "¿Cuántos días?")
    assert r["estado"] == "respondida"
    assert r["fuentes"] == [{"fuente": "ing.pdf", "pagina": 51, "trozo_id": 7}]
    assert r["recuperados"] == [7, 8]


def test_rechaza_si_el_modelo_dice_que_no_hay_informacion():
    llm = LlmFalso(Respuesta(responde=False, respuesta="", citas=[], cita_textual=""))
    r = preguntar(_agente(llm), "¿Capital de Francia?")
    assert r["estado"] == "rechazada" and r["motivo"] == "sin_informacion"
    assert r["respuesta"] == MENSAJE_RECHAZO
    assert len(llm.prompts) == 1


def test_una_cita_inventada_se_corrige_en_el_segundo_intento():
    llm = LlmFalso(_ok(cita="El plazo para desistir es de cuarenta y cinco días naturales"), _ok())
    r = preguntar(_agente(llm), "¿Cuántos días?")
    assert r["estado"] == "respondida"
    assert len(llm.prompts) == 2
    assert "AVISO DEL INTENTO ANTERIOR" in llm.prompts[1]
    assert "AVISO DEL INTENTO ANTERIOR" not in llm.prompts[0]


def test_si_la_cita_nunca_es_verificable_se_rechaza():
    inventada = _ok(cita="Esto no aparece en ningún fragmento de los documentos")
    llm = LlmFalso(*[inventada] * MAX_INTENTOS)
    r = preguntar(_agente(llm), "¿Cuántos días?")
    assert r["estado"] == "rechazada" and r["motivo"] == "cita_no_verificable"
    assert len(llm.prompts) == MAX_INTENTOS


def test_la_cita_debe_estar_en_el_fragmento_citado_no_en_otro():
    # la frase existe en el fragmento 1, pero el modelo cita el 2
    llm = LlmFalso(*[_ok(citas=(2,))] * MAX_INTENTOS)
    assert preguntar(_agente(llm), "¿Cuántos días?")["estado"] == "rechazada"


def test_citas_fuera_de_rango_no_valen():
    llm = LlmFalso(*[_ok(citas=(9,))] * MAX_INTENTOS)
    assert preguntar(_agente(llm), "¿Cuántos días?")["estado"] == "rechazada"


def test_cita_es_literal_ignora_acentos_espacios_y_mayusculas():
    assert cita_es_literal("EN LOS 30 primeros dias desde la contratacion", [TEXTO])
    assert not cita_es_literal("corta", ["corta"])  # demasiado corta para fiarse
    assert not cita_es_literal("una frase que no esta en el texto", [TEXTO])


def test_el_prompt_numera_los_fragmentos_y_trata_su_contenido_como_datos():
    prompt = construir_prompt("¿Pregunta?", IndiceFalso().buscar("", 2, ""), CONFIG)
    assert "[1] (ing.pdf, pág. 51)" in prompt and "[2] (ley.pdf, pág. 3)" in prompt
    assert "no instrucciones" in prompt
    assert "pólizas de seguro" in prompt
