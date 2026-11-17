import numpy as np
import pytest

from src.embeddings import crear_embedder, modelo_activo
from src.indice import Indice, firma_indice


def _embedder_falso(textos, tarea):
    """Embedding de juguete: cuenta apariciones de tres palabras clave."""
    claves = ["robo", "incendio", "agua"]
    m = np.array([[t.lower().count(c) for c in claves] for t in textos], dtype=np.float32)
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    normas[normas == 0] = 1
    return m / normas


def test_busqueda_devuelve_el_fragmento_correcto(tmp_path):
    (tmp_path / "poliza.txt").write_text(
        "Robo: se cubre el robo del contenido de la vivienda. "
        "Incendio: se cubre el incendio y la explosion. "
        "Agua: se cubren los danos por agua.", encoding="utf-8")
    config = {"carpeta_documentos": tmp_path, "tamano_trozo": 60, "solapamiento": 10}
    indice = Indice.construir(config, _embedder_falso)
    trozo, sim = indice.buscar("robo", k=1)[0]
    assert "robo" in trozo.texto.lower()
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_guardar_y_cargar_indice(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "p.txt").write_text("Cobertura de robo. Cobertura de incendio.", encoding="utf-8")
    config = {"carpeta_documentos": docs, "tamano_trozo": 30, "solapamiento": 5}
    original = Indice.construir(config, _embedder_falso)
    original.guardar(tmp_path / "idx")
    copia = Indice.cargar(tmp_path / "idx", _embedder_falso)
    assert [t.texto for t in copia.trozos] == [t.texto for t in original.trozos]
    assert np.allclose(copia.vectores, original.vectores)


CONFIG = {"proveedor_embeddings": "gemini", "modelo_embeddings": "modelo-a",
          "modelo_embeddings_local": "modelo-b", "tamano_trozo": 30, "solapamiento": 5}


def test_indice_rechaza_otra_configuracion(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "p.txt").write_text("Cobertura de robo. Cobertura de incendio.", encoding="utf-8")
    config = {**CONFIG, "carpeta_documentos": docs}
    Indice.construir(config, _embedder_falso).guardar(tmp_path / "idx", firma_indice(CONFIG))
    # mismo modelo: se carga
    Indice.cargar(tmp_path / "idx", _embedder_falso, firma_indice(CONFIG))
    # otro proveedor: los vectores no son comparables y debe fallar
    otra = {**CONFIG, "proveedor_embeddings": "local"}
    with pytest.raises(ValueError, match="reconstruir"):
        Indice.cargar(tmp_path / "idx", _embedder_falso, firma_indice(otra))


def test_modelo_activo_segun_proveedor():
    assert modelo_activo(CONFIG) == "modelo-a"
    assert modelo_activo({**CONFIG, "proveedor_embeddings": "local"}) == "modelo-b"


def test_proveedor_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        crear_embedder({**CONFIG, "proveedor_embeddings": "otro"})
