import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

image_directory = Path(__file__).resolve().parent / "promos"
image_paths = sorted(image_directory.glob("*.jpeg"))

if not image_paths:
    print(f"No .jpeg files found in {image_directory}")

for image_path in image_paths:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe the contents of this image in 3 bullet points, "
                    "using only the information mentioned inside it. Do not "
                    "describe the design. Only describe the promotion."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    )

    response = llm.invoke([message])
    print(f"\n--- {image_path.name} ---")
    print(response.content)
