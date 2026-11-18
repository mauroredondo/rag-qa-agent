
from src.evaluacion import auc, cargar_preguntas, es_acierto, intervalo_wilson
from src.troceado import Trozo, trocear_pagina


def test_es_acierto_exige_la_fuente_y_la_frase():
    pregunta = {"fuente": "a.pdf", "evidencia": ["plazo  maximo de siete dias"]}
    assert es_acierto(Trozo(0, "a.pdf", 1, "El PLAZO maximo de\nsiete dias."), pregunta)
    assert not es_acierto(Trozo(0, "b.pdf", 1, "plazo maximo de siete dias"), pregunta)
    assert not es_acierto(Trozo(0, "a.pdf", 1, "otra cosa"), pregunta)


def test_wilson_contiene_la_proporcion_y_se_estrecha_con_mas_datos():
    lo, hi = intervalo_wilson(20, 28)
    assert lo < 20 / 28 < hi
    lo2, hi2 = intervalo_wilson(200, 280)
    assert (hi2 - lo2) < (hi - lo)
    assert intervalo_wilson(0, 0) == (0.0, 0.0)


def test_auc():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5


def test_preguntas_del_repositorio_estan_bien_formadas():
    preguntas = cargar_preguntas()
    ids = [p["id"] for p in preguntas]
    assert len(ids) == len(set(ids))
    for p in preguntas:
        assert p["tipo"] in ("con_respuesta", "sin_respuesta")
        if p["tipo"] == "con_respuesta":
            assert p["fuente"] and p["evidencia"] and p["respuesta_esperada"]
        else:
            assert not p["fuente"] and not p["evidencia"]


def test_troceado_por_estructura_no_mezcla_articulos():
    texto = ("Articulo uno. " + "a" * 300 + " Articulo dos. " + "b" * 300)
    plano = trocear_pagina(texto, 900, 100)
    por_articulos = trocear_pagina(texto, 900, 100, separador=r"Articulo \w+\. ")
    assert len(plano) == 1  # cabe todo junto y se mezclan
    assert len(por_articulos) == 2
    assert por_articulos[0].startswith("Articulo uno") and por_articulos[1].startswith("Articulo dos")


def test_secciones_muy_cortas_se_unen_a_la_siguiente():
    texto = "Articulo uno. corto. Articulo dos. " + "b" * 300
    trozos = trocear_pagina(texto, 900, 100, separador=r"Articulo \w+\. ", minimo=200)
    assert len(trozos) == 1
