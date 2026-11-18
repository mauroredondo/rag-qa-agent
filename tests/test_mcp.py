import asyncio

from mcp.client import Client
from mcp.types import TextContent

from src.mcp_server import MAX_K, MAX_PREGUNTA, crear_servidor
from src.troceado import Trozo

CONFIG = {"k": 3, "modo_busqueda": "hibrido", "descripcion": "pólizas de seguro"}


class IndiceFalso:
    def __init__(self):
        self.trozos = [Trozo(0, "a.pdf", 1, "texto a1"), Trozo(1, "a.pdf", 2, "texto a2"),
                       Trozo(2, "b.pdf", 1, "texto b1")]
        self.llamadas = []

    def buscar(self, pregunta, k, modo):
        self.llamadas.append((pregunta, k, modo))
        return [(t, 0.5) for t in self.trozos[:k]]


class AgenteFalso:
    def invoke(self, entrada):
        return {"resultado": {"estado": "respondida", "respuesta": f"resp a {entrada['pregunta']}",
                              "cita_textual": "c", "fuentes": [], "recuperados": []}}


def _llamar(servidor, nombre, argumentos=None):
    async def ir():
        async with Client(servidor) as cliente:
            return await cliente.call_tool(nombre, argumentos or {})
    return asyncio.run(ir())


def _texto(resultado):
    """Texto del primer bloque de la respuesta (puede ser texto, imagen, audio...)."""
    bloque = resultado.content[0]
    assert isinstance(bloque, TextContent)
    return bloque.text


def _herramientas(servidor):
    async def ir():
        async with Client(servidor) as cliente:
            return (await cliente.list_tools()).tools
    return asyncio.run(ir())


def test_sin_agente_solo_se_publican_las_herramientas_de_busqueda():
    nombres = {t.name for t in _herramientas(crear_servidor(IndiceFalso(), CONFIG))}
    assert nombres == {"buscar_en_documentos", "listar_documentos"}


def test_con_agente_se_publica_tambien_preguntar():
    nombres = {t.name for t in _herramientas(crear_servidor(IndiceFalso(), CONFIG, AgenteFalso()))}
    assert "preguntar_a_documentos" in nombres


def test_todas_las_herramientas_son_de_solo_lectura():
    for t in _herramientas(crear_servidor(IndiceFalso(), CONFIG, AgenteFalso())):
        assert t.annotations is not None
        assert t.annotations.read_only_hint is True
        assert t.annotations.destructive_hint is False


def test_buscar_devuelve_fuente_pagina_y_texto_con_el_modo_del_config():
    indice = IndiceFalso()
    r = _llamar(crear_servidor(indice, CONFIG), "buscar_en_documentos", {"pregunta": "  hola  ", "k": 2})
    assert not r.is_error
    items = r.structured_content["result"]
    assert [i["fuente"] for i in items] == ["a.pdf", "a.pdf"]
    assert items[0]["pagina"] == 1 and items[0]["texto"] == "texto a1"
    assert indice.llamadas == [("hola", 2, "hibrido")]  # la pregunta se limpia


def test_k_se_limita_a_un_maximo():
    indice = IndiceFalso()
    _llamar(crear_servidor(indice, CONFIG), "buscar_en_documentos", {"pregunta": "x", "k": 9999})
    assert indice.llamadas[0][1] == MAX_K
    _llamar(crear_servidor(indice, CONFIG), "buscar_en_documentos", {"pregunta": "x", "k": -5})
    assert indice.llamadas[1][1] == 1


def test_preguntas_vacias_o_enormes_se_rechazan_sin_buscar():
    indice = IndiceFalso()
    servidor = crear_servidor(indice, CONFIG)
    vacia = _llamar(servidor, "buscar_en_documentos", {"pregunta": "   "})
    assert vacia.is_error and "no puede estar vacía" in _texto(vacia)  # mensaje claro
    larga = _llamar(servidor, "buscar_en_documentos", {"pregunta": "x" * (MAX_PREGUNTA + 1)})
    assert larga.is_error and "demasiado larga" in _texto(larga)
    assert indice.llamadas == []


def test_listar_documentos_cuenta_los_fragmentos():
    r = _llamar(crear_servidor(IndiceFalso(), CONFIG), "listar_documentos")
    assert r.structured_content["result"] == [{"fuente": "a.pdf", "fragmentos": 2},
                                              {"fuente": "b.pdf", "fragmentos": 1}]


def test_preguntar_devuelve_el_resultado_del_agente():
    r = _llamar(crear_servidor(IndiceFalso(), CONFIG, AgenteFalso()), "preguntar_a_documentos",
                {"pregunta": "¿Plazo?"})
    assert r.structured_content["estado"] == "respondida"
    assert r.structured_content["respuesta"] == "resp a ¿Plazo?"
