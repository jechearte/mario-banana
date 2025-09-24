# CLI de generación de trazados de karting

Esta herramienta de línea de comandos genera imágenes del trazado de un circuito de karting a partir de una imagen aérea usando la API de Gemini. El flujo principal consiste en enviar la imagen de referencia, recibir varias propuestas y seleccionar automáticamente la mejor según unos criterios definidos.

## Requisitos
- Python 3.10 o superior.
- Dependencias indicadas en `requirements.txt`.
- Cuenta en Google AI Studio y clave activa de la API de Gemini.

## Instalación
1. Crea y activa un entorno virtual (opcional, pero recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Configuración
Copia el archivo `.env.example` a `.env` (si no existe, créalo) y define al menos:
   ```env
   GEMINI_API_KEY="tu_clave_de_gemini"
   ```

## Uso básico
Ejecuta el script `generate_image.py` indicando la imagen aérea:
```bash
python generate_image.py /ruta/a/imagen_aerea.png
```

### Argumentos opcionales
- `-d, --description`: añade contexto textual sobre el circuito.
- `--no-copy`: evita copiar la imagen original a la carpeta del proyecto.
- `--num`: cantidad de peticiones paralelas (por defecto 10).
- `--price-in` y `--price-out`: sobrescriben el cálculo de costes por tokens.

Si no pasas la ruta de la imagen, el programa te pedirá que arrastres o pegues el archivo en la terminal.

## Salida generada
- Cada ejecución crea una carpeta `images/run_<timestamp>` con:
  - Copia de la imagen original (si no usas `--no-copy`).
  - Imágenes candidatas generadas por Gemini.
  - `winner_<timestamp>_<archivo>`: copia de la imagen seleccionada como mejor resultado.
  - `run_summary.json`: resumen con rutas, tokens consumidos y estimación de costes.

## Carpeta `test_images`
Incluye pares de imágenes de ejemplo (`imagen_aerea_*` y `trazado_*`) que alimentan a Gemini como few-shot examples. No la borres ni la ignores para mantener la calidad de las respuestas.

## Errores comunes
- **Falta de clave**: si no se encuentra `GEMINI_API_KEY`, el programa termina mostrando instrucciones para configurarla.
- **Ruta inválida**: al arrastrar/pegar un archivo inexistente, se solicitará nuevamente hasta recibir una ruta válida.

## Desarrollo
- El código principal está en `generate_image.py`, donde se gestiona la carga de la imagen, el envío de peticiones concurrentes a Gemini y la selección automática del resultado óptimo.
- Los archivos generados en `images/` están excluidos del repositorio mediante `.gitignore` para mantener el repo ligero.

## Próximos pasos sugeridos
- Añadir tests automáticos para validar la lógica de selección.
- Automatizar la creación de un ejemplo de `.env` en el repositorio.

