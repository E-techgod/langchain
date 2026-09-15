import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class Promotion(BaseModel):
    Insurance_Provided: str = Field(
        description="The insurance provider named in the promotion."
    )
    Promo_date: str = Field(
        description="The promotion date or date range exactly as shown."
    )
    Promo_info: str = Field(
        description="A short description of the promotion or offer."
    )
    Restrictions: list[str] | None = Field(
        default=None,
        description="Any restrictions, exclusions, or conditions shown.",
    )

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
structured_llm = llm.with_structured_output(Promotion)

image_directory = Path(__file__).resolve().parent / "promos"
processed_file = Path(__file__).resolve().parent / "processed_images.txt"
image_paths = sorted(image_directory.glob("*.jpeg"))
processed_images = (
    set(processed_file.read_text(encoding="utf-8").splitlines())
    if processed_file.exists()
    else set()
)
unprocessed_paths = [
    image_path for image_path in image_paths if image_path.name not in processed_images
]

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

    response = structured_llm.invoke([message])
    print(f"\n--- {image_path.name} ---")
    print(response.model_dump_json(indent=2))

    with processed_file.open("a", encoding="utf-8") as file:
        file.write(f"{image_path.name}\n")
