#!/usr/bin/env python3
"""Local web app: recognize numbers in a photo and compare two arrays."""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    from PIL import Image, ImageOps
except ImportError:  # Show a useful error instead of failing at startup.
    Image = None
    ImageOps = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

MAX_NUMBERS = 54
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
NUMBER_PATTERN = re.compile(r"\d+")


def canonical_number(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def extract_numbers(text: str) -> list[str]:
    values: list[str] = []
    for token in NUMBER_PATTERN.findall(text):
        try:
            values.append(canonical_number(Decimal(token)))
        except InvalidOperation:
            continue
    return values


def parse_array(text: str, label: str) -> list[str]:
    values = extract_numbers(text)
    if not values:
        raise ValueError(f"В массиве {label} не найдено чисел от 0 до 99.")
    if any(Decimal(value) > 99 for value in values):
        raise ValueError(f"В массиве {label} допустимы только целые числа от 0 до 99.")
    if len(values) > MAX_NUMBERS:
        raise ValueError(f"В массиве {label} найдено {len(values)} чисел. Максимум — {MAX_NUMBERS}.")
    return values


def is_lotto_ticket(image) -> tuple[bool, int]:
    """Recognize the red-logo Russian Lotto card and return its grid's left edge."""
    width, height = image.size
    ratio = width / height
    if not 1.35 <= ratio <= 2.5:
        return False, 0

    # Analyze a small preview so large phone photos stay cheap to inspect.
    preview = image.copy()
    preview.thumbnail((500, 500), Image.Resampling.LANCZOS)
    pw, ph = preview.size
    probe = preview.convert("RGB")
    logo_right = -1
    red_pixels = 0
    left_limit = int(pw * 0.32)
    top_limit = int(ph * 0.52)
    for y in range(top_limit):
        for x in range(left_limit):
            red, green, blue = probe.getpixel((x, y))
            if red > 125 and red > green * 1.35 and red > blue * 1.25:
                red_pixels += 1
                logo_right = max(logo_right, x)

    region_size = max(1, left_limit * top_limit)
    if red_pixels / region_size < 0.008 or logo_right < 0:
        return False, 0

    # Skip the logo panel and the printed ticket number. The grid begins just
    # to the right of the red logo on this ticket layout.
    crop_fraction = logo_right / pw + 0.05
    crop_left = min(int(width * 0.38), max(int(width * 0.20), int(width * crop_fraction)))
    return True, crop_left


def prepare_image(image_bytes: bytes) -> tuple[bytes, bool]:
    if Image is None:
        raise RuntimeError("Не установлена библиотека Pillow. Выполните: python -m pip install -r requirements.txt")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            ticket_mode, grid_left = is_lotto_ticket(image)
            if ticket_mode:
                image = image.crop((grid_left, 0, image.width, image.height))
                # The number cells on a shared ticket photo can be small.
                # Enlarging the cropped grid gives OCR more pixels per digit.
                scale = min(3.0, 1100 / max(image.size))
                if scale > 1:
                    image = image.resize(
                        (int(image.width * scale), int(image.height * scale)),
                        Image.Resampling.LANCZOS,
                    )
                    image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
            max_pixels = 20_000_000
            if image.width * image.height > max_pixels:
                scale = (max_pixels / (image.width * image.height)) ** 0.5
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            if max(image.size) > 3500:
                scale = 3500 / max(image.size)
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            for _ in range(8):
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=90, optimize=True)
                encoded = output.getvalue()
                if len(encoded) <= 9 * 1024 * 1024:
                    return encoded, ticket_mode
                image = image.resize(
                    (int(image.width * 0.9), int(image.height * 0.9)),
                    Image.Resampling.LANCZOS,
                )
            raise ValueError("Фото слишком большое. Уменьшите его размер и загрузите снова.")
    except Exception as exc:
        raise ValueError("Не удалось открыть изображение. Загрузите фото в формате JPG, PNG или WEBP.") from exc


def recognize(image_bytes: bytes) -> tuple[str, list[str], bool]:
    if Image is None or pytesseract is None:
        raise RuntimeError("Локальный OCR не установлен. Пересоберите приложение через Docker.")
    prepared, ticket_mode = prepare_image(image_bytes)
    try:
        with Image.open(io.BytesIO(prepared)) as image:
            image_width, image_height = image.size
            data = pytesseract.image_to_data(
                image,
                lang="eng",
                config="--psm 11 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
                timeout=25,
            )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("На сервере не найден Tesseract OCR. Пересоберите приложение через Docker.") from exc
    except RuntimeError as exc:
        raise RuntimeError("Локальный OCR не смог обработать фото. Попробуйте более чёткий снимок.") from exc
    except Exception as exc:
        raise RuntimeError(f"Ошибка локального OCR: {exc}") from exc

    recognized_words: list[str] = []
    if ticket_mode:
        # The Russian Lotto grid has 9 columns and 6 rows. Group OCR tokens by
        # their cell so even sparse-text OCR stays in the ticket's reading order.
        cell_width, cell_height = image_width / 9, image_height / 6
        cells: dict[tuple[int, int], list[tuple[int, str]]] = {}
        for index, text in enumerate(data.get("text", [])):
            tokens = NUMBER_PATTERN.findall(str(text))
            if not tokens:
                continue
            left = int(data["left"][index])
            top = int(data["top"][index])
            token_width = int(data["width"][index])
            token_height = int(data["height"][index])
            row = max(0, min(5, int((top + token_height / 2) / cell_height)))
            column = max(0, min(8, int((left + token_width / 2) / cell_width)))
            for token in tokens:
                if len(token) <= 2:
                    cells.setdefault((row, column), []).append((left, token))
        for row in range(6):
            for column in range(9):
                pieces = sorted(cells.get((row, column), []))
                number = "".join(piece for _, piece in pieces)
                if number and len(number) <= 2:
                    recognized_words.append(number)
    else:
        order = sorted(
            range(len(data.get("text", []))),
            key=lambda index: (int(data["top"][index]), int(data["left"][index])),
        )
        for index in order:
            recognized_words.extend(NUMBER_PATTERN.findall(str(data["text"][index])))

    values = [canonical_number(Decimal(value)) for value in recognized_words]
    recognized_text = ", ".join(values)
    if any(Decimal(value) > 99 for value in values):
        raise ValueError(
            "Распознано число больше 99. Допустимы только однозначные и двузначные числа. "
            "Проверьте, что на фото нет посторонних чисел, и переснимите массив крупнее."
        )
    if len(values) > MAX_NUMBERS:
        raise ValueError(f"На фото найдено {len(values)} чисел. Максимум — {MAX_NUMBERS}; проверьте фото.")
    return recognized_text, values, ticket_mode


def compare_arrays(m1: list[str], m2: list[str], mode: str) -> dict:
    first = [Decimal(value) for value in m1]
    second = [Decimal(value) for value in m2]
    first_counts, second_counts = Counter(first), Counter(second)
    common = first_counts & second_counts
    common_display = [
        {"value": canonical_number(value), "count": count}
        for value, count in sorted(common.items(), key=lambda item: first.index(item[0]))
    ]
    missing = first_counts - second_counts
    extra = second_counts - first_counts

    def list_difference(source: list[Decimal], difference: Counter) -> list[str]:
        seen: Counter = Counter()
        result: list[str] = []
        for value in source:
            seen[value] += 1
            if seen[value] <= difference[value]:
                result.append(canonical_number(value))
        return result

    missing_values = list_difference(first, missing)
    extra_values = list_difference(second, extra)
    positional_matches = sum(a == b for a, b in zip(first, second))
    count = positional_matches if mode == "position" else sum(common.values())
    return {
        "matched_count": count,
        "by_value_count": sum(common.values()),
        "positional_count": positional_matches,
        "m1_count": len(first),
        "m2_count": len(second),
        "matched_values": common_display,
        "missing_from_photo": missing_values,
        "extra_on_photo": extra_values,
        "mode": mode,
    }


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Сравнение массивов чисел</title>
  <style>
    :root { color-scheme: light; --ink:#172b45; --muted:#637b94; --line:#d8e6f2; --blue:#2563eb; --pale:#eaf7ff; --red:#a9364a; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f3f8fc; color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { width:min(900px, calc(100% - 32px)); margin:42px auto 64px; }
    h1 { margin:0 0 8px; font-size:clamp(28px,5vw,40px); letter-spacing:-.035em; }
    .intro { margin:0 0 28px; color:var(--muted); }
    .card { background:white; border:1px solid var(--line); border-radius:18px; padding:24px; margin:16px 0; box-shadow:0 6px 20px #173b6010; }
    h2 { font-size:20px; margin:0 0 8px; }
    .hint { color:var(--muted); margin:0 0 16px; font-size:14px; }
    label { display:block; font-weight:650; margin:0 0 8px; }
    textarea { display:block; width:100%; min-height:112px; padding:13px 14px; border:1px solid #cadbec; border-radius:11px; resize:vertical; font:15px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--ink); }
    textarea:focus, input:focus { outline:3px solid #2563eb25; border-color:var(--blue); }
    .row { display:flex; flex-wrap:wrap; align-items:center; gap:12px; }
    button { border:0; border-radius:10px; padding:11px 17px; background:linear-gradient(135deg,#2563eb,#0891b2); color:#fff; font-weight:700; font-size:15px; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    .file-input { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
    .camera-button { display:inline-flex; align-items:center; justify-content:center; border-radius:10px; padding:12px 18px; background:linear-gradient(135deg,#2563eb,#0891b2); color:#fff; font-weight:700; cursor:pointer; }
    .camera-button:focus-within, .file-input:focus + .camera-button { outline:3px solid #2563eb25; outline-offset:2px; }
    .camera-button.disabled { opacity:.55; cursor:wait; pointer-events:none; }
    .secondary { background:#e7f4ff; color:#14508f; }
    .message { min-height:24px; margin:10px 0 0; color:var(--muted); font-size:14px; }
    .error { color:var(--red); }
    .mode { border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin:14px 0 18px; }
    .mode label { display:inline-flex; align-items:center; gap:8px; margin:4px 22px 4px 0; font-weight:500; cursor:pointer; }
    .result { display:none; background:var(--pale); border:1px solid #c6e8f7; border-radius:16px; padding:18px 20px; margin-top:18px; }
    .result.visible { display:block; }
    .count { font-size:48px; line-height:1.1; color:var(--blue); font-weight:800; letter-spacing:-.04em; }
    .count-label { color:#275982; font-weight:650; }
    details { margin-top:18px; border-top:1px solid #d1e7f4; padding-top:12px; }
    summary { cursor:pointer; font-weight:700; }
    .details-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin-top:12px; }
    .details-grid p { margin:4px 0 0; overflow-wrap:anywhere; color:#405970; }
    .foot { margin-top:20px; color:var(--muted); font-size:13px; }
    @media(max-width:560px) { main{margin:24px auto 40px}.card{padding:18px}.row button{width:100%} }
  </style>
</head>
<body>
<main>
  <h1>Сравнение массивов чисел</h1>
  <p class="intro">Введите исходный массив 1 текстом или с фотографии, затем загрузите фото массива 2.</p>

  <section class="card">
    <h2>1. Массив 1</h2>
    <p class="hint">До 54 целых чисел от 0 до 99. Введите их через запятую или распознайте с фотографии; билет «Русское лото» тоже можно загрузить целиком. Массив 1 сохраняется в этом браузере.</p>
    <label for="m1">Числа массива 1</label>
    <textarea id="m1" placeholder="Например: 12, 7, 4, 18, 7"></textarea>
    <div class="row" style="margin-top:12px">
      <input id="m1Photo" class="file-input" type="file" accept="image/*" capture="environment">
      <label id="m1CameraButton" class="camera-button" for="m1Photo">Сфотографировать или загрузить массив 1</label>
    </div>
    <p id="m1OcrMessage" class="message" aria-live="polite"></p>
  </section>

  <section class="card">
    <h2>2. Массив 2</h2>
    <p class="hint">Нажмите кнопку, чтобы открыть камеру или выбрать фото. Можно загрузить билет с числами в клетках: приложение обрежет логотип и номер билета, а затем распознает числа в сетке. Снимок загрузится автоматически; результат можно поправить вручную.</p>
    <div class="row">
      <input id="photo" class="file-input" type="file" accept="image/*" capture="environment">
      <label id="cameraButton" class="camera-button" for="photo">Сфотографировать или загрузить массив 2</label>
    </div>
    <p id="ocrMessage" class="message" aria-live="polite"></p>
    <label for="m2">Числа массива 2</label>
    <textarea id="m2" placeholder="Например: 12, 7, 4, 18, 7"></textarea>
  </section>

  <section class="card">
    <h2>3. Сравнение</h2>
    <fieldset class="mode">
      <legend>Как считать совпадения</legend>
      <label><input type="radio" name="mode" value="unordered" checked> Без учёта порядка (повторы учитываются)</label>
      <label><input type="radio" name="mode" value="position"> Только одинаковые позиции</label>
    </fieldset>
    <button id="compare" type="button">Сравнить массивы</button>
    <p id="compareMessage" class="message" aria-live="polite"></p>
    <div id="result" class="result" aria-live="polite">
      <div class="count" id="count">0</div>
      <div class="count-label" id="countLabel">совпадений</div>
      <p id="summary"></p>
      <details>
        <summary>Показать отдельное сравнение</summary>
        <div class="details-grid">
          <div><strong>Совпавшие значения</strong><p id="matched"></p></div>
          <div><strong>Есть в массиве 1, нет в массиве 2</strong><p id="missing"></p></div>
          <div><strong>Есть в массиве 2, нет в массиве 1</strong><p id="extra"></p></div>
          <div><strong>Совпадения по позициям</strong><p id="positions"></p></div>
        </div>
      </details>
    </div>
  </section>
  <p class="foot">Для билета «Русское лото» приложение распознаёт поле с клетками и пропускает логотип с номером билета. Распознавание выполняется на сервере, без отправки фото во внешний OCR-сервис; проверьте распознанные числа перед сравнением.</p>
</main>
<script>
  const byId = id => document.getElementById(id);
  function showMessage(id, text, isError=false) {
    const el = byId(id); el.textContent = text; el.classList.toggle('error', isError);
  }
  try { byId('m1').value = localStorage.getItem('arrayCompare.m1') || ''; } catch (_) {}
  byId('m1').addEventListener('input', () => {
    try { localStorage.setItem('arrayCompare.m1', byId('m1').value); } catch (_) {}
  });
  async function recognizeFile(file) {
    const form = new FormData(); form.append('photo', file);
    const response = await fetch('/api/recognize', {method:'POST', body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось распознать фото.');
    return data;
  }
  function recognitionMessage(data, label) {
    const subject = data.ticket ? 'В сетке билета' : `В массиве ${label}`;
    const ticketNote = data.ticket ? ' Номер билета исключён.' : '';
    return `${subject} найдено чисел: ${data.numbers.length}.${ticketNote} Проверьте результат и при необходимости исправьте его.`;
  }
  byId('m1Photo').addEventListener('change', async () => {
    const file = byId('m1Photo').files[0];
    if (!file) return;
    const input = byId('m1Photo'); const button = byId('m1CameraButton');
    input.disabled = true; button.classList.add('disabled'); button.textContent = 'Распознаю массив 1…';
    showMessage('m1OcrMessage', 'Обрабатываю фото массива 1…');
    try {
      const data = await recognizeFile(file);
      byId('m1').value = data.numbers.join(', ');
      try { localStorage.setItem('arrayCompare.m1', byId('m1').value); } catch (_) {}
      showMessage('m1OcrMessage', recognitionMessage(data, '1'));
      if (byId('m2').value.trim()) await compareArrays();
      else showMessage('compareMessage', 'Теперь загрузите фото массива 2.');
    } catch (error) { showMessage('m1OcrMessage', error.message, true); }
    finally { input.disabled = false; button.classList.remove('disabled'); button.textContent = 'Сфотографировать или загрузить массив 1'; input.value = ''; }
  });
  byId('photo').addEventListener('change', async () => {
    const file = byId('photo').files[0];
    if (!file) return;
    const input = byId('photo'); const button = byId('cameraButton');
    input.disabled = true; button.classList.add('disabled'); button.textContent = 'Распознаю массив 2…';
    showMessage('ocrMessage', 'Обрабатываю фото массива 2…');
    try {
      const data = await recognizeFile(file);
      byId('m2').value = data.numbers.join(', ');
      showMessage('ocrMessage', recognitionMessage(data, '2'));
      if (byId('m1').value.trim()) await compareArrays();
      else showMessage('compareMessage', 'Введите или распознайте массив 1, чтобы посчитать совпадения.');
    } catch (error) { showMessage('ocrMessage', error.message, true); }
    finally { input.disabled = false; button.classList.remove('disabled'); button.textContent = 'Сфотографировать или загрузить массив 2'; input.value = ''; }
  });
  async function compareArrays() {
    const button = byId('compare'); button.disabled = true;
    showMessage('compareMessage', 'Считаю совпадения…'); byId('result').classList.remove('visible');
    try {
      const mode = document.querySelector('input[name="mode"]:checked').value;
      const response = await fetch('/api/compare', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({m1:byId('m1').value,m2:byId('m2').value,mode})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Не удалось сравнить массивы.');
      byId('count').textContent = data.matched_count;
      byId('countLabel').textContent = data.matched_count === 1 ? 'совпадение' : (data.matched_count >= 2 && data.matched_count <= 4 ? 'совпадения' : 'совпадений');
      byId('summary').textContent = `В массиве 1: ${data.m1_count}; в массиве 2: ${data.m2_count}. Совпадений без учёта порядка: ${data.by_value_count}.`;
      byId('matched').textContent = data.matched_values.length ? data.matched_values.map(item => `${item.value} × ${item.count}`).join(', ') : 'Нет';
      byId('missing').textContent = data.missing_from_photo.length ? data.missing_from_photo.join(', ') : 'Нет';
      byId('extra').textContent = data.extra_on_photo.length ? data.extra_on_photo.join(', ') : 'Нет';
      byId('positions').textContent = `${data.positional_count} совпадений из ${Math.min(data.m1_count, data.m2_count)} сравниваемых позиций`;
      byId('result').classList.add('visible'); showMessage('compareMessage', '');
    } catch (error) { showMessage('compareMessage', error.message, true); }
    finally { button.disabled = false; }
  }
  byId('compare').addEventListener('click', compareArrays);
</script>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ArrayCompare/1.0"

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Некорректный размер запроса.") from exc
        if length <= 0:
            raise ValueError("Запрос пустой.")
        if length > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise ValueError("Файл слишком большой. Максимальный размер — 15 МБ.")
        return self.rfile.read(length)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self.read_body()
            if path == "/api/recognize":
                if len(body) > MAX_UPLOAD_BYTES + 1024 * 1024:
                    raise ValueError("Файл слишком большой. Максимальный размер — 15 МБ.")
                message = BytesParser(policy=default).parsebytes(
                    b"Content-Type: " + self.headers.get("Content-Type", "").encode("ascii", errors="ignore")
                    + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
                )
                photo = next((part for part in message.iter_parts() if part.get_filename()), None)
                if photo is None:
                    raise ValueError("Не найден загруженный файл.")
                image_bytes = photo.get_payload(decode=True) or b""
                if len(image_bytes) > MAX_UPLOAD_BYTES:
                    raise ValueError("Файл слишком большой. Максимальный размер — 15 МБ.")
                raw_text, numbers, ticket_mode = recognize(image_bytes)
                if not numbers:
                    raise ValueError("На фото не найдены числа от 0 до 99. Снимите числа или билет крупно, резко и без наклона, или введите массив вручную.")
                self.send_json({"numbers": numbers, "recognized_text": raw_text, "ticket": ticket_mode})
                return
            if path == "/api/compare":
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Некорректные данные для сравнения.") from exc
                m1 = parse_array(str(payload.get("m1", "")), "1")
                m2 = parse_array(str(payload.get("m2", "")), "2")
                mode = payload.get("mode", "unordered")
                if mode not in ("unordered", "position"):
                    raise ValueError("Выберите корректный режим сравнения.")
                self.send_json(compare_arrays(m1, m2, mode))
                return
            self.send_error(404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, 503)
        except Exception as exc:
            self.log_error("Unexpected request error: %s", exc)
            self.send_json({"error": "Внутренняя ошибка приложения. Попробуйте ещё раз."}, 500)

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный сайт для распознавания и сравнения массивов чисел")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес для запуска (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Порт (по умолчанию 8000)")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Сайт запущен: http://{args.host}:{args.port}")
    print("Остановить сервер: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
