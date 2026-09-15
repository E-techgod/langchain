import base64
import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from guardrails import Guard

from promotion_schema import Promotion
from llm import llm

structured_llm = llm.with_structured_output(Promotion)
lcel_fallback_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
lcel_fallback = RunnableLambda(lambda message: [message]) | lcel_fallback_llm | StrOutputParser()
promotion_guard = Guard.for_pydantic(output_class=Promotion)

image_directory = Path(__file__).resolve().parent / "promos"
processed_file = Path(__file__).resolve().parent / "processed_images.txt"
results_file = Path(__file__).resolve().parent / "promotion_results.json"
image_paths = sorted(image_directory.glob("*.jpeg"))
processed_images = (
    set(processed_file.read_text(encoding="utf-8").splitlines())
    if processed_file.exists()
    else set()
)
promotion_results = (
    json.loads(results_file.read_text(encoding="utf-8"))
    if results_file.exists()
    else []
)
unprocessed_paths = [
    image_path for image_path in image_paths if image_path.name not in processed_images
]


def parse_json_response(raw_response: str) -> Promotion:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return Promotion.model_validate_json(cleaned.strip())


def validate_with_guardrails(promotion: Promotion) -> Promotion:
    validation = promotion_guard.validate(promotion.model_dump_json())
    if not validation.validation_passed:
        raise ValueError("Guardrails rejected the extracted promotion")
    return Promotion.model_validate(validation.validated_output)


def extract_promotion(message: HumanMessage) -> Promotion:
    try:
        response = structured_llm.invoke([message])
        return validate_with_guardrails(response)
    except Exception as structured_error:
        print(f"Structured extraction failed; trying LCEL fallback: {structured_error}")

    try:
        raw_response = lcel_fallback.invoke(message)
        promotion = parse_json_response(raw_response)
        return validate_with_guardrails(promotion)
    except Exception as lcel_error:
        raise RuntimeError(
            "Both structured extraction and the LCEL/Guardrails fallback failed"
        ) from lcel_error

if not image_paths:
    print(f"No .jpeg files found in {image_directory}")
elif not unprocessed_paths:
    print("No new .jpeg files to process.")

for image_path in unprocessed_paths:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extract the promotion details using only information "
                    "visible in the image. Do not describe the design. If no "
                    "restrictions are shown, return null for Restrictions."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    )

    response = extract_promotion(message)
    print(f"\n--- {image_path.name} ---")
    print(response.model_dump_json(indent=2))

    promotion_results.append(
        {
            "filename": image_path.name,
            **response.model_dump(),
        }
    )
    # Python—not the LLM—serializes the validated results and writes the file.
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(promotion_results, file, indent=2, ensure_ascii=False)
        file.write("\n")

    with processed_file.open("a", encoding="utf-8") as file:
        file.write(f"{image_path.name}\n")
