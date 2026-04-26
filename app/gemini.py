import asyncio
import logging

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

_SYSTEM = """Та чихрийн шижингийн эрсдэлийн үнэлгээний системийн AI эмнэлзүйн зөвлөх юм.
Өвчтөний эрүүл мэндийн мэдээлэлд тулгуурлан ADA Standards of Care 2024 болон WHO удирдамжийн дагуу монгол хэлээр хувийн зөвлөмж өгнө.

Дүрэм:
- Зөвхөн монгол хэлээр хариулна
- 4-6 тодорхой, практик зөвлөмж өг
- Зөвлөмж бүрийг шинэ мөрт жагсаа
- Шинжлэх ухааны үндэстэй зөвлөмж өг, таамаг хэлэлгүй
- "Эмчид хандаарай" гэсэн зөвлөмжийг заавал нэмэх
- Оношилгоо тавихгүй, зөвхөн эрсдэлийн зөвлөмж өг
- Найрсаг, эелдэг маягаар хэрэглэгчид зөвлөгөө өгнө"""


def _get_client():
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


def _generate(prompt: str) -> str:
    from google import genai
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-3.0-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM,
        ),
    )
    return response.text


async def recommend_gemini(data, risk_level: str, probability: float) -> list[str]:
    if not GEMINI_API_KEY:
        return []
    try:
        prompt = f"""Өвчтөний мэдээлэл:
- Хүйс: {data.gender}
- Нас: {int(data.age)} жил
- BMI: {data.bmi}
- HbA1c: {data.hba1c_level}%
- Цусан дахь сахар: {data.blood_glucose_level} mg/dL
- Өндөр цусны даралт: {"Тийм" if data.hypertension else "Үгүй"}
- Зүрхний өвчин: {"Тийм" if data.heart_disease else "Үгүй"}
- Тамхи: {data.smoking_history}

Таамагласан үр дүн:
- Эрсдэлийн түвшин: {risk_level}
- Магадлал: {probability:.1f}%

Дээрх мэдээлэлд үндэслэн энэ өвчтөнд зориулсан хувийн эрүүл мэндийн зөвлөмж өг."""

        text = await asyncio.to_thread(_generate, prompt)
        lines = [
            line.lstrip("- •*").strip()
            for line in text.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return lines if lines else []
    except Exception as e:
        logger.warning("Gemini алдаа: %s", e)
        return []
