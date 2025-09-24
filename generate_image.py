#!/usr/bin/env python3
"""
CLI para generar imágenes usando la API de Gemini
Genera una imagen del trazado de un circuito de karting con fondo transparente
a partir de una imagen aérea del circuito. Uso por terminal (arrastrar archivo).
"""

import argparse
import mimetypes
import os
import shutil
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Cargar variables de entorno desde .env en la raíz del proyecto
base_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(base_dir, ".env"))
IMAGES_DIR = os.path.join(base_dir, "images")
# Identificador de ejecución para agrupar entradas/salidas
EXECUTION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(IMAGES_DIR, f"run_{EXECUTION_ID}")

# Precios por 1M tokens (por defecto los que indicaste)
DEFAULT_PRICE_IN_PER_1M = float(os.environ.get("GEMINI_PRICE_IN", 0.30))
DEFAULT_PRICE_OUT_PER_1M = float(os.environ.get("GEMINI_PRICE_OUT", 30.0))
IMAGES_DIR = os.path.join(base_dir, "images")


def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def save_binary_file(file_name, data):
    """Guarda datos binarios en un archivo"""
    ensure_dir(os.path.dirname(file_name) or ".")
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"Imagen guardada en: {file_name}")


def read_image_file(image_path):
    """
    Lee un archivo de imagen y devuelve sus bytes y el tipo MIME
    
    Args:
        image_path (str): Ruta al archivo de imagen
        
    Returns:
        tuple: (data_bytes, mime_type) o (None, None) si hay error
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Detectar tipo MIME
        mime_type = mimetypes.guess_type(image_path)[0]
        if not mime_type or not mime_type.startswith('image/'):
            # Intentar detectar por extensión
            ext = os.path.splitext(image_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')

        return image_data, mime_type
    except Exception as e:
        print(f"Error al leer la imagen: {str(e)}")
        return None, None


def _normalize_key(name: str) -> str:
    name = unicodedata.normalize("NFC", name)
    name = name.lower()
    name = name.replace(" ", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def collect_example_pairs(max_pairs: int = 3):
    """Empareja archivos de ejemplo en test_images por sufijo común.

    Busca archivos con prefijo 'imagen_aerea_' y 'trazado_' y los empareja por el sufijo
    (tras normalización), p.ej.: imagen_aerea_los santos.* ↔ trazado_los_santos.*
    """
    examples_dir = os.path.join(base_dir, "test_images")
    if not os.path.isdir(examples_dir):
        return []

    aerial_map = {}
    trace_map = {}
    for fname in os.listdir(examples_dir):
        fpath = os.path.join(examples_dir, fname)
        if not os.path.isfile(fpath):
            continue
        root, ext = os.path.splitext(fname)
        if not ext.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if root.startswith("imagen_aerea_"):
            key = _normalize_key(root[len("imagen_aerea_"):])
            aerial_map[key] = fpath
        elif root.startswith("trazado_"):
            key = _normalize_key(root[len("trazado_"):])
            trace_map[key] = fpath

    pairs = []
    for key in aerial_map.keys() & trace_map.keys():
        pairs.append((aerial_map[key], trace_map[key]))
        if len(pairs) >= max_pairs:
            break
    return pairs


def copy_image_to_project(image_path: str, run_dir: str | None = None) -> str:
    """
    Copia la imagen al directorio del proyecto con un nombre único.

    Args:
        image_path (str): Ruta original de la imagen

    Returns:
        str: Ruta de destino dentro del proyecto
    """
    try:
        if not os.path.isfile(image_path):
            print("Error: La ruta especificada no es un archivo válido")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(image_path)
        target_dir = run_dir or RUN_DIR
        ensure_dir(target_dir)
        dest_name = f"input_{timestamp}_{base_name}"
        dest_path = os.path.join(target_dir, dest_name)
        shutil.copy2(image_path, dest_path)
        print(f"Imagen de entrada copiada a: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"No se pudo copiar la imagen al proyecto: {str(e)}")
        return ""


def generate_circuit_image(image_path, additional_description="", index: int | None = None, price_in: float | None = None, price_out: float | None = None, run_dir: str | None = None):
    """
    Genera una imagen del trazado de un circuito usando Gemini API
    
    Args:
        image_path (str): Ruta a la imagen aérea del circuito
        additional_description (str): Descripción adicional del circuito
    """
    # Verificar que existe la API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: No se encontró la variable de entorno GEMINI_API_KEY")
        print("Por favor, configura tu API key de Gemini:")
        print("export GEMINI_API_KEY='tu_api_key_aqui'")
        return False
    
    # Leer la imagen
    image_data, mime_type = read_image_file(image_path)
    if not image_data:
        print("Error: No se pudo leer la imagen")
        return False
    
    # Crear cliente de Gemini
    client = genai.Client(api_key=api_key)
    
    # Modelo para generar imágenes
    model = "gemini-2.5-flash-image-preview"
    
    # Construcción de contenido con ejemplos + entrada del usuario
    user_prompt_text = """Generate the image of this circuit layout.
The image should only show the track layout, filled in black; everything outside the track must not be visible. 

IMPORTANT:

- Curbs on the corners must not be visible in the layout image.
- The background of the image must be white.
    """

    contents = []

    # Recoger hasta 3 pares de ejemplo (imagen_aerea[_i] + trazado[_i])
    example_pairs = collect_example_pairs(max_pairs=3)
    for aerial_path, trace_path in example_pairs:
        ex_img_bytes, ex_img_mime = read_image_file(aerial_path)
        if ex_img_bytes:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=user_prompt_text),
                        types.Part.from_bytes(mime_type=ex_img_mime, data=ex_img_bytes),
                    ],
                )
            )
        ex_tr_bytes, ex_tr_mime = read_image_file(trace_path)
        if ex_tr_bytes:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_bytes(mime_type=ex_tr_mime, data=ex_tr_bytes)],
                )
            )

    # Mensaje final del usuario: prompt con anti-caché + imagen del usuario + info adicional
    request_uuid = str(uuid.uuid4())
    print(f"uuid:{request_uuid}")
    no_cache_prefix = f"[no-cache: {request_uuid}]"
    user_parts = [
        types.Part.from_text(text=no_cache_prefix + user_prompt_text),
        types.Part.from_bytes(mime_type=mime_type, data=image_data),
    ]
    if additional_description.strip():
        user_parts.append(
            types.Part.from_text(text=f"Información adicional: {additional_description}")
        )
    contents.append(types.Content(role="user", parts=user_parts))

    try:
        print(
            f"Contenido preparado → mensajes={len(contents)} | user_parts_final={len(user_parts)}"
        )
    except Exception:
        pass
    
    # Configuración para generar contenido
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_modalities=[
            "IMAGE"
        ],
        system_instruction=[
            types.Part.from_text(text="""Eres un agente que genera imágenes de trazados de circuitos de karting."""),
        ]
    )
    
    try:
        print("Generando imagen del circuito...")
        
        # Generar contenido (sin streaming)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Buscar imagen en la respuesta
        image_saved = False
        # Tokens/usage
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_in": 0.0,
            "cost_out": 0.0,
            "cost_total": 0.0,
        }
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Si encontramos datos inline (imagen)
                if part.inline_data and part.inline_data.data:
                    # Generar nombre de archivo con timestamp
                    target_dir = run_dir or RUN_DIR
                    ensure_dir(target_dir)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    suffix = f"_{index}" if index is not None else ""
                    file_name = f"output_{timestamp}{suffix}"
                    
                    # Obtener extensión del archivo basada en el tipo MIME
                    file_extension = mimetypes.guess_extension(part.inline_data.mime_type)
                    if not file_extension:
                        file_extension = ".png"  # Por defecto PNG
                    
                    full_file_name = os.path.join(target_dir, f"{file_name}{file_extension}")
                    
                    # Guardar imagen
                    save_binary_file(full_file_name, part.inline_data.data)
                    image_saved = True
                
                # Si hay texto en la respuesta, mostrarlo
                elif hasattr(part, 'text') and part.text:
                    print(f"Respuesta del modelo: {part.text}")
        # Extraer usage/tokens
        try:
            usage_md = getattr(response, 'usage_metadata', None) or getattr(response, 'usage', None)
            if usage_md:
                usage["input_tokens"] = int(getattr(usage_md, 'prompt_token_count', 0) or 0)
                usage["output_tokens"] = int(getattr(usage_md, 'candidates_token_count', 0) or 0)
                usage["total_tokens"] = int(getattr(usage_md, 'total_token_count', usage["input_tokens"] + usage["output_tokens"]))
                # Calcular costes
                pin = price_in if price_in is not None else DEFAULT_PRICE_IN_PER_1M
                pout = price_out if price_out is not None else DEFAULT_PRICE_OUT_PER_1M
                usage["cost_in"] = (usage["input_tokens"] / 1_000_000.0) * pin
                usage["cost_out"] = (usage["output_tokens"] / 1_000_000.0) * pout
                usage["cost_total"] = usage["cost_in"] + usage["cost_out"]
        except Exception:
            pass
        
        if not image_saved:
            print("No se pudo extraer ninguna imagen de la respuesta")
            return False
            
        print("¡Imagen generada exitosamente!")
        return True, full_file_name if image_saved else None, usage
        
    except Exception as e:
        print(f"Error al generar la imagen: {str(e)}")
        return False, None, {"error": str(e)}


def _parse_json_from_text(text: str):
    """Intenta extraer un objeto JSON de un texto arbitrario."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        return None
    return None


def select_winner_image(candidate_image_paths, price_in: float | None = None, price_out: float | None = None, reference_image_path: str | None = None):
    """Envía las imágenes candidatas a Gemini para elegir una ganadora.

    Devuelve (ok: bool, selection: dict, usage: dict)
    selection contiene winner_index (int|None), reason (str), confidence (float|None)
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: No se encontró la variable de entorno GEMINI_API_KEY")
            return False, {"error": "missing_api_key"}, {"error": "missing_api_key"}

        if not candidate_image_paths:
            return False, {"error": "no_candidates"}, {}

        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"

        instruction_text = (
            "A un modelo de IA se le ha pedido generar a partir de la vista aérea de un circuito de karting una imagen de su trazado."
            "El modelo de IA ha generado varias imágenes"
            "Tu misión es seleccionar la imagen ganadora, para ello debes evaluar los siguentes criterios de selección:"
            "1) Debe mostrar únicamente el trazado, relleno en negro, sin elementos extra.\n"
            "2) Fidelidad a la forma general del trazado.\n\n"
            "3) Fondo completamente blanco.\n"
            "4) No deben aparecer pianos en las curvas.\n"
            "Además de las imágenes generadas por la IA también recibirás la imagen aérea original\n\n"
            "Siempre debes seleccionar una imagen ganadora."
        )

        parts = [types.Part.from_text(text=instruction_text)]
        # Añadir imagen de referencia (aérea) si está disponible
        if reference_image_path:
            ref_bytes, ref_mime = read_image_file(reference_image_path)
            if ref_bytes:
                parts.append(types.Part.from_text(text="Imagen de referencia (aérea)"))
                parts.append(types.Part.from_bytes(mime_type=ref_mime, data=ref_bytes))
        for idx, img_path in enumerate(candidate_image_paths):
            img_bytes, img_mime = read_image_file(img_path)
            if not img_bytes:
                continue
            parts.append(types.Part.from_text(text=f"Candidata {idx}"))
            parts.append(types.Part.from_bytes(mime_type=img_mime, data=img_bytes))

        contents = [types.Content(role="user", parts=parts)]

        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(
                thinking_budget=512,
            ),
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["winner_index", "confidence", "reason"],
                properties={
                    "winner_index": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "confidence": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "reason": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                },
            ),
            system_instruction=[
                types.Part.from_text(text="Eres un evaluador objetivo que solo devuelve JSON válido."),
            ],
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_in": 0.0,
            "cost_out": 0.0,
            "cost_total": 0.0,
        }

        parsed = None
        raw_text = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    raw_text = (raw_text or "") + str(part.text)
        if raw_text:
            parsed = _parse_json_from_text(raw_text)

        try:
            usage_md = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
            if usage_md:
                usage["input_tokens"] = int(getattr(usage_md, "prompt_token_count", 0) or 0)
                usage["output_tokens"] = int(getattr(usage_md, "candidates_token_count", 0) or 0)
                usage["total_tokens"] = int(getattr(usage_md, "total_token_count", usage["input_tokens"] + usage["output_tokens"]))
                pin = price_in if price_in is not None else DEFAULT_PRICE_IN_PER_1M
                pout = price_out if price_out is not None else DEFAULT_PRICE_OUT_PER_1M
                usage["cost_in"] = (usage["input_tokens"] / 1_000_000.0) * pin
                usage["cost_out"] = (usage["output_tokens"] / 1_000_000.0) * pout
                usage["cost_total"] = usage["cost_in"] + usage["cost_out"]
        except Exception:
            pass

        if not isinstance(parsed, dict):
            return False, {"error": "invalid_response", "raw_text": raw_text}, usage

        winner_index = parsed.get("winner_index")
        # Convertir winner_index string -> int o None
        if isinstance(winner_index, str):
            if winner_index.strip().lower() in ("", "null", "none", "-1"):
                winner_index = None
            else:
                try:
                    winner_index = int(winner_index)
                except Exception:
                    winner_index = None
        elif winner_index is not None:
            try:
                winner_index = int(winner_index)
            except Exception:
                winner_index = None
        reason = parsed.get("reason")
        confidence = parsed.get("confidence")
        # Convertir confidence string -> float
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except Exception:
                confidence = None
        else:
            try:
                confidence = None if confidence is None else float(confidence)
            except Exception:
                confidence = None

        selection = {
            "winner_index": winner_index if (isinstance(winner_index, int) and 0 <= winner_index < len(candidate_image_paths)) else None,
            "reason": reason if isinstance(reason, str) else None,
            "confidence": confidence,
            "raw_text": raw_text,
        }

        return True, selection, usage
    except Exception as e:
        return False, {"error": str(e)}, {"error": str(e)}


def normalize_path_from_terminal(raw_path: str) -> str:
    """Normaliza rutas pegadas/arrastradas en la terminal.

    - Elimina comillas envolventes
    - Convierte file:// URLs a ruta local
    - Desescapa espacios con backslash
    - Expande ~ y convierte a absoluta
    """
    if not raw_path:
        return ""

    path = raw_path.strip()

    # Quitar comillas simples o dobles al inicio/fin
    if (path.startswith("'") and path.endswith("'")) or (path.startswith('"') and path.endswith('"')):
        path = path[1:-1]

    # file:// URL → ruta local
    if path.startswith("file://"):
        parsed = urlparse(path)
        path = unquote(parsed.path)

    # Desescapar espacios estilo shell (/My\ Folder/My\ File.png)
    path = path.replace("\\ ", " ")

    # Expandir ~ y normalizar
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    return path


def main():
    """Entrada de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera una imagen del trazado de un circuito con fondo transparente "
            "a partir de una imagen aérea. Puedes arrastrar el archivo a la terminal."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        help=(
            "Ruta a la imagen aérea del circuito (PNG/JPG/WEBP...). "
            "En macOS/Linux puedes arrastrar el archivo a la terminal para pegar la ruta."
        ),
    )
    parser.add_argument(
        "-d",
        "--description",
        default="",
        help="Descripción adicional del circuito (opcional)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="No copiar la imagen al proyecto; usar la ruta original",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Número de peticiones paralelas",
    )
    parser.add_argument(
        "--price-in",
        type=float,
        default=DEFAULT_PRICE_IN_PER_1M,
        help="Precio por 1M tokens de entrada (override)",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=DEFAULT_PRICE_OUT_PER_1M,
        help="Precio por 1M tokens de salida (override)",
    )

    args = parser.parse_args()

    # Si no se pasó argumento, pedirlo por consola (drag & drop o pegar ruta)
    if not args.image:
        print("Arrastra aquí la imagen aérea del circuito o pega su ruta y presiona Enter:")
        while True:
            raw = input("> ").strip()
            image_path = normalize_path_from_terminal(raw)
            if image_path and os.path.isfile(image_path):
                break
            print("Ruta inválida. Vuelve a arrastrar o pega una ruta válida.")
    else:
        image_path = normalize_path_from_terminal(args.image)
        if not os.path.isfile(image_path):
            print("Ruta inválida. Asegúrate de que el archivo existe.")
            raise SystemExit(1)

    # Copiar al proyecto salvo que se indique lo contrario
    if args.no_copy:
        local_image_path = image_path
    else:
        copied_path = copy_image_to_project(image_path)
        local_image_path = copied_path or image_path

    # Lanzar N peticiones en paralelo con cálculo de costes
    num = max(1, int(args.num))
    print(f"Lanzando {num} peticiones en paralelo...")
    results = []
    with ThreadPoolExecutor(max_workers=num) as ex:
        futs = [
            ex.submit(
                generate_circuit_image,
                local_image_path,
                args.description,
                i + 1,
                args.price_in,
                args.price_out,
            )
            for i in range(num)
        ]
        for f in as_completed(futs):
            try:
                results.append(f.result())  # (success, path, usage)
            except Exception as e:
                print(f"Error en tarea: {e}")
                results.append((False, None, {"error": str(e)}))

    # Agregación de tokens/costes y escritura de resumen
    total_in = 0
    total_out = 0
    total_cost_in = 0.0
    total_cost_out = 0.0
    successes = 0
    output_items = []
    for ok, path, usage in results:
        if ok:
            successes += 1
        if isinstance(usage, dict):
            total_in += int(usage.get("input_tokens", 0) or 0)
            total_out += int(usage.get("output_tokens", 0) or 0)
            total_cost_in += float(usage.get("cost_in", 0.0) or 0.0)
            total_cost_out += float(usage.get("cost_out", 0.0) or 0.0)
        output_items.append({"ok": ok, "path": path, "usage": usage})

    total_cost = total_cost_in + total_cost_out

    # Selección de imagen ganadora con gemini-2.5-flash
    candidate_paths = [it["path"] for it in output_items if it.get("ok") and it.get("path")]
    selection_info = None
    selection_usage = None
    winner_path = None
    if candidate_paths:
        print("Solicitando a Gemini la selección de la imagen ganadora...")
        sel_ok, selection_info, selection_usage = select_winner_image(
            candidate_paths, price_in=args.price_in, price_out=args.price_out, reference_image_path=local_image_path
        )
        if isinstance(selection_usage, dict):
            total_in += int(selection_usage.get("input_tokens", 0) or 0)
            total_out += int(selection_usage.get("output_tokens", 0) or 0)
            total_cost_in += float(selection_usage.get("cost_in", 0.0) or 0.0)
            total_cost_out += float(selection_usage.get("cost_out", 0.0) or 0.0)
            total_cost = total_cost_in + total_cost_out
        # Mostrar razonamiento/explicación de Gemini
        if isinstance(selection_info, dict):
            try:
                print("Razonamiento de Gemini (selección):")
                print(f" - winner_index: {selection_info.get('winner_index')}")
                print(f" - confidence: {selection_info.get('confidence')}")
                print(f" - reason: {selection_info.get('reason')}")
            except Exception:
                pass
        if sel_ok and isinstance(selection_info, dict):
            win_idx = selection_info.get("winner_index")
            if isinstance(win_idx, int) and 0 <= win_idx < len(candidate_paths):
                src = candidate_paths[win_idx]
                dir_name = os.path.dirname(src)
                target_dir = dir_name if dir_name else RUN_DIR
                ensure_dir(target_dir)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = os.path.basename(src)
                winner_path = os.path.join(target_dir, f"winner_{ts}_{base}")
                try:
                    shutil.copy2(src, winner_path)
                    print(f"Imagen ganadora copiada a: {winner_path}")
                except Exception as e:
                    print(f"No se pudo copiar la imagen ganadora: {e}")

    ensure_dir(RUN_DIR)
    summary_path = os.path.join(RUN_DIR, "run_summary.json")
    summary = {
        "model": "gemini-2.5-flash-image-preview",
        "execution_id": EXECUTION_ID,
        "run_dir": RUN_DIR,
        "prices_per_1M": {"input": args.price_in, "output": args.price_out},
        "num_requests": num,
        "results": output_items,
        "selection": {
            "model": "gemini-2.5-flash",
            "info": selection_info,
            "usage": selection_usage,
            "winner_path": winner_path,
        },
        "totals": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_in": round(total_cost_in, 6),
            "cost_out": round(total_cost_out, 6),
            "cost_total": round(total_cost, 6),
        },
    }
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Resumen guardado en: {summary_path}")
    except Exception as e:
        print(f"No se pudo guardar el resumen: {e}")

    print(
        f"Tokens totales → input: {total_in} | output: {total_out}. Coste → in: {total_cost_in:.4f}€, out: {total_cost_out:.4f}€, total: {total_cost:.4f}€."
    )

    success = successes > 0
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
