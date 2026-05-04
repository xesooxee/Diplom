"""
Translate data/mongolian_foods.csv food names to Mongolian.

This keeps the original `name` column as the internal backend/model key and
adds/fills `name_mn` for UI display.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "mongolian_foods.csv"
BACKUP_PATH = BASE_DIR / "data" / "mongolian_foods.before_name_mn.csv"


SYSTEM_PROMPT = """You translate food ingredient names from English to Mongolian for a Mongolian food/nutrition app.
Return only valid JSON. No markdown.

Rules:
- Use common Mongolian Cyrillic ingredient names that a Mongolian user would understand.
- Prefer names used in Mongolian cooking and recipes, for example: гурил, будаа, мах, сүү, тараг, өндөг, төмс, лууван, сонгино, байцаа, тос.
- Prefer the most common everyday Mongolian ingredient name over literal USDA-style translation.
- If a detailed English name is uncommon in Mongolia, simplify it to the closest familiar ingredient name.
- Examples:
  - "Rye flour, light" -> "Гурил" or "Хөх тарианы гурил"
  - "Flour, wheat, all-purpose" -> "Гурил"
  - "Beef, raw, ground" -> "Татсан үхрийн мах"
  - "Rice, white, cooked" -> "Болгосон цагаан будаа"
  - "Pasta, dry" -> "Гоймон"
  - "Cream, sour" -> "Зөөхий"
- Make the name useful as a food ingredient label, not a long scientific database label.
- Keep names short, natural, and clear for UI display.
- Preserve important preparation/quality words in Mongolian when relevant: raw=түүхий, cooked=болгосон, boiled=чанасан, roasted=шарсан, dry=хуурай, frozen=хөлдөөсөн, low fat=өөх тос багатай, fat free=өөх тосгүй, light=цайвар/хөнгөн.
- Do not over-translate technical descriptors such as enriched, unenriched, separable, trimmed, choice, select, grade. Drop them unless they are important for normal food selection.
- Reorder words into natural Mongolian word order.
- Do not include English words. If a brand name appears, remove the brand and translate the actual food type.
- If the item is too brand-specific, return a generic Mongolian ingredient name such as "Жигнэмэг", "Талх", "Сүү", "Тараг", "Ундаа", or "Хүнсний бүтээгдэхүүн".
- Return a JSON object where each original English name maps to one Mongolian display name.
"""


def get_client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY олдсонгүй. .env файл эсвэл environment шалгана уу.")
    return genai.Client(api_key=api_key)


def parse_json_object(text: str) -> dict[str, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"JSON object олдсонгүй: {text[:200]}")
    return json.loads(cleaned[start:end + 1])


def translate_batch(client, names: list[str], model: str) -> dict[str, str]:
    from google import genai

    prompt = (
        "Translate these food names to Mongolian. Return JSON object only.\n"
        + json.dumps(names, ensure_ascii=False)
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    data = parse_json_object(response.text or "")
    return {
        name: str(data.get(name, "")).strip()
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all missing names.")
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")

    df = pd.read_csv(CSV_PATH)
    if "name" not in df.columns:
        raise RuntimeError("CSV дотор name багана алга.")
    if "name_mn" not in df.columns:
        df.insert(1, "name_mn", "")

    missing_mask = df["name_mn"].fillna("").astype(str).str.strip().eq("")
    missing_names = df.loc[missing_mask, "name"].astype(str).tolist()
    if args.limit:
        missing_names = missing_names[:args.limit]

    print(f"Нийт мөр: {len(df)}", flush=True)
    print(f"Орчуулах шаардлагатай: {len(missing_names)}", flush=True)
    if not missing_names:
        df.to_csv(CSV_PATH, index=False)
        print("Бүх мөр name_mn утгатай байна.", flush=True)
        return

    if not BACKUP_PATH.exists():
        df.to_csv(BACKUP_PATH, index=False)
        print(f"Backup үүсгэлээ: {BACKUP_PATH}", flush=True)

    client = get_client()
    translated_count = 0
    failures: list[str] = []

    for start in range(0, len(missing_names), args.batch_size):
        batch = missing_names[start:start + args.batch_size]
        batch_no = start // args.batch_size + 1
        total_batches = (len(missing_names) + args.batch_size - 1) // args.batch_size
        print(f"[{batch_no}/{total_batches}] translating {len(batch)} names...", flush=True)

        try:
            translations = translate_batch(client, batch, args.model)
        except Exception as exc:
            print(f"  batch failed: {exc}", flush=True)
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                print("Quota дууссан тул зогслоо. Дараа дахин ажиллуулахад хоосон name_mn мөрүүдээс үргэлжилнэ.", flush=True)
                break
            failures.extend(batch)
            continue

        for original, translated in translations.items():
            if not translated:
                failures.append(original)
                continue
            df.loc[df["name"].astype(str).eq(original), "name_mn"] = translated
            translated_count += 1

        df.to_csv(CSV_PATH, index=False)
        print(f"  saved. translated so far: {translated_count}", flush=True)
        time.sleep(args.sleep)

    print(f"Дууслаа. Орчуулсан: {translated_count}. Амжилтгүй: {len(failures)}", flush=True)
    if failures:
        fail_path = BASE_DIR / "data" / "food_name_translation_failures.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2))
        print(f"Амжилтгүй нэрс: {fail_path}", flush=True)


if __name__ == "__main__":
    main()
