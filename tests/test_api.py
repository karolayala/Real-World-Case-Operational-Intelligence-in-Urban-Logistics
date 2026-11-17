"""Pruebas de los endpoints de la API."""


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_aplicacion_web(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "Estimador de entregas" in r.text


def test_opciones(cliente):
    res = cliente.get("/opciones").json()
    assert "Low" in res["categorias"]["Traffic_Level"]
    assert "Distance_km" in res["variables_usadas"]


def test_prediccion_normal(cliente, pedido):
    r = cliente.post("/predict", json=pedido)
    assert r.status_code == 200
    res = r.json()
    assert res["limite_inferior_min"] <= res["tiempo_estimado_min"] <= res["limite_superior_min"]
    assert res["alertas"] == []


def test_contribuciones_suman_la_prediccion(cliente, pedido):
    res = cliente.post("/predict", json=pedido).json()
    total = res["base_min"] + sum(c["minutos"] for c in res["contribuciones"])
    assert abs(total - res["tiempo_estimado_min"]) < 0.05


def test_faltantes_se_imputan(cliente, pedido):
    r = cliente.post("/predict", json={**pedido, "Weather": None, "Courier_Experience_yrs": None})
    assert r.status_code == 200
    assert "Weather imputada" in r.json()["alertas"]
    assert "Courier_Experience_yrs imputada" in r.json()["alertas"]


def test_categoria_nueva(cliente, pedido):
    r = cliente.post("/predict", json={**pedido, "Weather": "Hail"})
    assert r.status_code == 200
    assert "Weather imputada" in r.json()["alertas"]


def test_fuera_de_rango(cliente, pedido):
    r = cliente.post("/predict", json={**pedido, "Distance_km": 45})
    assert any("Distance_km fuera de rango" in a for a in r.json()["alertas"])


def test_distancia_negativa_rechazada(cliente, pedido):
    assert cliente.post("/predict", json={**pedido, "Distance_km": -5}).status_code == 422


def test_texto_en_numerico_rechazado(cliente, pedido):
    assert cliente.post("/predict", json={**pedido, "Distance_km": "lejos"}).status_code == 422


def test_lote(cliente, pedido):
    lote = [{**pedido, "id": "A"}, {**pedido, "id": "B", "Distance_km": 2}, {"id": "C"}]
    res = cliente.post("/predict/batch", json=lote).json()
    assert [x["id"] for x in res] == ["A", "B", "C"]
    assert res[1]["tiempo_estimado_min"] < res[0]["tiempo_estimado_min"]
    assert len(res[2]["alertas"]) > 0


def test_lote_coincide_con_individual(cliente, pedido):
    individual = cliente.post("/predict", json=pedido).json()["tiempo_estimado_min"]
    lote = cliente.post("/predict/batch", json=[pedido]).json()[0]["tiempo_estimado_min"]
    assert individual == lote


def test_lote_vacio(cliente):
    assert cliente.post("/predict/batch", json=[]).json() == []
