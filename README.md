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

Errores: `422` por campos o coordenadas inválidas, puntos iguales o extensión superior a 3 grados; `404` si la malla no permite una ruta; `503` si Copernicus no está disponible o falla la adquisición de datos. Los detalles dependen del resultado real del proveedor.

## Resultados de verificación reales

El `curl` real Cancún → Playa del Carmen respondió `200`. Copernicus reutilizó un NetCDF real que ya estaba en caché. Los dos puntos se ajustaron a celdas marinas, con desplazamientos de 10.54 km y 5.24 km:

```json
{"route":{"type":"LineString","coordinates":[[-86.75,21.166672],[-86.75,21.083336],[-86.75,21.0],[-86.75,20.916672],[-86.833328,20.833336],[-86.833328,20.75],[-86.916664,20.666672],[-87.0,20.583336],[-87.083328,20.583336]]},"meta":{"distance_km":83.801,"time_minutes":377.6,"data_source":"Copernicus Marine (archivo local reutilizado)","selected_time":"2026-11-21T00:00:00.000000000","snap_warnings":["El punto 'Origen (A)' (21.161900, -86.851500) cayó sobre una celda terrestre o sin datos del modelo. Se ajustó al nodo marino más cercano (21.166672, -86.750000), desplazamiento de 10538.4 m (10.54 km). Este ajuste depende de la resolución del modelo y NO garantiza un punto de salida o navegación segura.","El punto 'Destino (B)' (20.629600, -87.073900) cayó sobre una celda terrestre o sin datos del modelo. Se ajustó al nodo marino más cercano (20.583336, -87.083328), desplazamiento de 5237.1 m (5.24 km). Este ajuste depende de la resolución del modelo y NO garantiza un punto de salida o navegación segura."],"disclaimer":"AVISO: Esta ruta ha sido calculada con fines de demostración para un MVP utilizando datos oceanográficos de resolución media y sin cartas náuticas, batimetría, arrecifes ni avisos a los marinos. NO es una ruta validada ni segura para navegación real."}}
```

Puerto Morelos también respondió `200` con una descarga real de Copernicus:

```json
{"route":{"type":"LineString","coordinates":[[-86.833328,20.833336],[-86.75,20.833336],[-86.666664,20.833336]]},"meta":{"distance_km":17.321,"time_minutes":62.14,"data_source":"Copernicus Marine (cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i)","selected_time":"2026-11-21T00:00:00.000000000","snap_warnings":["El punto 'Origen (A)' (20.841667, -86.878056) cayó sobre una celda terrestre o sin datos del modelo. Se ajustó al nodo marino más cercano (20.833336, -86.833328), desplazamiento de 4739.6 m (4.74 km). Este ajuste depende de la resolución del modelo y NO garantiza un punto de salida o navegación segura."],"disclaimer":"AVISO: Esta ruta ha sido calculada con fines de demostración para un MVP utilizando datos oceanográficos de resolución media y sin cartas náuticas, batimetría, arrecifes ni avisos a los marinos. NO es una ruta validada ni segura para navegación real."}}
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
