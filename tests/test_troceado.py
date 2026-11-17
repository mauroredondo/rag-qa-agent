import pytest

from src.carga import Pagina, cargar_documentos
from src.troceado import trocear_documentos, trocear_pagina

TEXTO = ("El asegurador cubre el robo con violencia en el domicilio. "
         "Quedan excluidos los danos causados por negligencia grave. ") * 10


def test_trozos_respetan_tamano_y_solapan():
    trozos = trocear_pagina(TEXTO, tamano=200, solapamiento=50)
    assert len(trozos) > 1
    assert all(len(t) <= 200 for t in trozos)
    # el final de un trozo reaparece al principio del siguiente
    assert trozos[0][-20:] in trozos[1]


def test_solapamiento_invalido():
    with pytest.raises(ValueError):
        trocear_pagina(TEXTO, tamano=100, solapamiento=100)


def test_texto_corto_da_un_solo_trozo():
    assert trocear_pagina("Hola mundo.", 500, 50) == ["Hola mundo."]


def test_trozos_conservan_fuente_y_pagina():
    paginas = [Pagina("a.pdf", 1, TEXTO), Pagina("b.pdf", 3, TEXTO)]
    trozos = trocear_documentos(paginas, 300, 50)
    assert {t.fuente for t in trozos} == {"a.pdf", "b.pdf"}
    assert [t.id for t in trozos] == list(range(len(trozos)))
    assert all(t.pagina == 3 for t in trozos if t.fuente == "b.pdf")


def test_cargar_txt(tmp_path):
    (tmp_path / "poliza.txt").write_text("Cobertura de incendio.", encoding="utf-8")
    (tmp_path / "ignorado.csv").write_text("x", encoding="utf-8")
    paginas = cargar_documentos(tmp_path)
    assert [p.fuente for p in paginas] == ["poliza.txt"]
