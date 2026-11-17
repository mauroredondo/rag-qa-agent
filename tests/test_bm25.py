import numpy as np
import pytest

from src.bm25 import BM25, tokenizar
from src.indice import Indice, fusion_rrf
from src.troceado import Trozo


def test_tokenizar_quita_acentos_plurales_y_palabras_vacias():
    assert tokenizar("¿Está cubierto el robo de las JOYAS?") == ["esta", "cubierto", "robo", "joya"]


def test_bm25_prefiere_el_fragmento_con_la_palabra_rara():
    textos = ["la prima se paga cada ano", "la franquicia es la parte que asume el asegurado",
              "la prima y la franquicia"]
    puntos = BM25(textos).puntuaciones("franquicia")
    assert puntos[1] > puntos[0] and puntos[2] > puntos[0]
    assert puntos[0] == 0


def test_bm25_penaliza_fragmentos_largos():
    textos = ["franquicia", "franquicia " + "relleno " * 50]
    puntos = BM25(textos).puntuaciones("franquicia")
    assert puntos[0] > puntos[1]


def test_fusion_rrf_premia_lo_que_aparece_en_las_dos_listas():
    fusion = fusion_rrf([[1, 2, 3], [3, 4, 1]])
    orden = [i for i, _ in fusion]
    assert set(orden[:2]) == {1, 3}  # estan en ambas listas
    assert orden[-1] in (2, 4)


def _indice(prefijo=False):
    trozos = [Trozo(0, "caser.pdf", 1, "cobertura de robo"),
              Trozo(1, "ing.pdf", 1, "cobertura de incendio"),
              Trozo(2, "asisa.pdf", 1, "periodo de carencia")]
    def embedder(textos, tarea):
        claves = ["robo", "incendio", "carencia"]
        m = np.array([[t.lower().count(c) for c in claves] for t in textos], dtype=np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True); n[n == 0] = 1
        return m / n
    return Indice(trozos, embedder([t.texto for t in trozos], None), embedder, prefijo)


@pytest.mark.parametrize("modo", ["denso", "bm25", "hibrido"])
def test_los_tres_modos_encuentran_el_fragmento(modo):
    trozo, _ = _indice().buscar("robo", k=1, modo=modo)[0]
    assert trozo.id == 0


def test_modo_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        _indice().buscar("robo", modo="otro")


def test_prefijo_fuente_permite_encontrar_por_nombre_del_documento():
    # la palabra 'ing' solo esta en el nombre del archivo, no en el texto
    sin = _indice(prefijo=False).buscar("ing", k=1, modo="bm25")[0][1]
    con = _indice(prefijo=True).buscar("ing", k=1, modo="bm25")[0]
    assert sin == 0
    assert con[0].fuente == "ing.pdf" and con[1] > 0
