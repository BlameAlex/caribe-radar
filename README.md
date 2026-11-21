# Caribe Radar API

API HTTP mínima para calcular una ruta marítima con A* y corrientes superficiales. Descarga datos reales de Copernicus Marine; la API no genera rutas con corrientes simuladas.

## Requisitos e instalación

Se requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --port 8000
```

Configura las credenciales con `copernicusmarine login` o las variables de entorno admitidas por Copernicus Marine. El archivo de credenciales suele quedar en `~/.copernicusmarine/.copernicusmarine-credentials`.

La caché de NetCDF se guarda en `data/cache/` y usa una clave derivada de la caja geográfica y la fecha solicitada. Los valores de `.env.example` permiten ajustar la velocidad del barco, margen de descarga, extensión máxima, refinamiento y directorio de caché.

## Endpoints

- `GET /health` comprueba que el servicio está activo.
- `POST /api/v1/routes` calcula la ruta. Swagger está disponible en `/docs`.

Solicitud:

```json
{
  "origin": {"lat": 21.1619, "lon": -86.8515},
  "destination": {"lat": 20.6296, "lon": -87.0739}
}
```

```sh
curl -s -X POST http://localhost:8000/api/v1/routes \
  -H 'Content-Type: application/json' \
  -d '{"origin":{"lat":21.1619,"lon":-86.8515},"destination":{"lat":20.6296,"lon":-87.0739}}'
```

La respuesta `200` contiene `route` como GeoJSON `LineString` y `meta` con distancia, tiempo estimado, fuente, instante seleccionado, avisos de ajuste a celdas marinas y aviso de seguridad. Las coordenadas GeoJSON se expresan en WGS84 en orden `[longitud, latitud]`. La ruta comienza y termina en las celdas marinas asignadas por el algoritmo, que pueden diferir de los puntos enviados.

La geometría se suaviza por defecto con una B-spline cúbica clamped, usando los puntos A* como controles y ocho muestras por segmento. Si la curva cae en tierra o celdas sin datos según la celda más cercana, se devuelve intacta la ruta A* y `meta.smoothing.applied` será `false` con el motivo. Se puede configurar con `SMOOTHING` y `SMOOTHING_SAMPLES`. `distance_km` y `time_minutes` siguen siendo las métricas de la ruta A* sin suavizar.

Errores: `422` por campos o coordenadas inválidas, puntos iguales o extensión superior a 3 grados; `404` si la malla no permite una ruta; `503` si Copernicus no está disponible o falla la adquisición de datos. Los detalles dependen del resultado real del proveedor.

## Resultados de verificación reales

El `curl` real Cancún → Playa del Carmen respondió `200`. Copernicus reutilizó un NetCDF real que ya estaba en caché. La respuesta aplicó el suavizado: nueve puntos A* produjeron 49 puntos, con los extremos anclados a las celdas marinas ajustadas (desplazamientos de 10.54 km y 5.24 km).

```json
{"route":{"type":"LineString","coordinates":[[-86.75,21.166672],[-86.74999999999997,21.137306371093743],[-86.75,21.111440218749998],[-86.75,21.08866664453125],[-86.75,21.068578749999997],[-86.75,21.050769636718748],[-86.75,21.034832406249997],[-86.75,21.02036016015625],[-86.75,21.006946],[-86.75002712499999,20.994237274739582],[-86.750217,20.982098322916666],[-86.750732375,20.97044773046875],[-86.75173600000001,20.959204083333336],[-86.753390625,20.948285967447916],[-86.755859,20.937611968749998],[-86.759303875,20.92710067317708],[-86.76388799999998,20.916670666666665],[-86.76969274999999,20.90625410677083],[-86.776474,20.895837437499996],[-86.78390625,20.885420674479164],[-86.79166399999997,20.875003833333327],[-86.79942175,20.864586929687494],[-86.806854,20.854169979166667],[-86.81363525,20.843752997395832],[-86.81944,20.833336000000003],[-86.82405125260416,20.822919002604163],[-86.82768602083334,20.812502020833332],[-86.83066982031251,20.802085070312504],[-86.83332816666666,20.791668166666664],[-86.83598657552083,20.781251325520834],[-86.83897056250001,20.770834562500003],[-86.84260564322919,20.76041789322917],[-86.84721733333333,20.75000133333333],[-86.85306333203125,20.73957132682292],[-86.86013007291665,20.72906003125],[-86.86833617317707,20.718386032552083],[-86.87760024999999,20.707467916666666],[-86.8878409205729,20.69622426953125],[-86.89897680208333,20.684573677083332],[-86.91092651171876,20.672434725260416],[-86.92360866666665,20.659726],[-86.9370232513021,20.646474605468747],[-86.95149571875001,20.63314171875],[-86.9674328893229,20.62029703515625],[-86.98524158333333,20.608510249999995],[-87.00532862109375,20.59835105859375],[-87.02810082291668,20.59038915625],[-87.05396500911459,20.585194238281254],[-87.083328,20.583336]]},"meta":{"distance_km":83.801,"time_minutes":377.6,"data_source":"Copernicus Marine (archivo local reutilizado)","selected_time":"2026-11-21T00:00:00.000000000","snap_warnings":["El punto 'Origen (A)' (21.161900, -86.851500) cayó sobre una celda terrestre o sin datos del modelo. Se ajustó al nodo marino más cercano (21.166672, -86.750000), desplazamiento de 10538.4 m (10.54 km). Este ajuste depende de la resolución del modelo y NO garantiza un punto de salida o navegación segura.","El punto 'Destino (B)' (20.629600, -87.073900) cayó sobre una celda terrestre o sin datos del modelo. Se ajustó al nodo marino más cercano (20.583336, -87.083328), desplazamiento de 5237.1 m (5.24 km). Este ajuste depende de la resolución del modelo y NO garantiza un punto de salida o navegación segura."],"disclaimer":"AVISO: Esta ruta ha sido calculada con fines de demostración para un MVP utilizando datos oceanográficos de resolución media y sin cartas náuticas, batimetría, arrecifes ni avisos a los marinos. NO es una ruta validada ni segura para navegación real.","smoothing":{"applied":true,"method":"cubic_bspline_clamped","raw_points":9,"reason":null}}}
```

`GET /docs` respondió `200` (`text/html`).

## Pruebas

```sh
uv run pytest -v
```

El test marcado `e2e` contacta Copernicus Marine si existe el archivo local de credenciales. Las pruebas funcionales restantes usan el generador sintético solo como fixture.

## Limitaciones

- La resolución del producto oceánico es aproximadamente 0.083° y puede no representar canales estrechos ni detalles costeros.
- El algoritmo ajusta origen y destino a celdas marinas válidas; esos nodos no son puntos náuticos verificados.
- El cálculo usa una instantánea superficial de corrientes y la trata como constante durante todo el trayecto.
- No incluye cartas náuticas, batimetría, arrecifes ni avisos a los marinos. **No es apto para navegación real.**
