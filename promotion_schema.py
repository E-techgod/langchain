from pydantic import BaseModel, Field


class Promotion(BaseModel):
    Insurance_Provided: str = Field(
        description="The insurance provider named in the promotion."
    )
    Promo_date: str = Field(
        description="The promotion date or date range exactly as shown."
    )
    Promo_info: list[str] = Field(
        description="A short description of the promotion or offer."
    )
    Restrictions: list[str] | None = Field(
        default=None,
        description="Any restrictions, exclusions, or conditions shown.",
    )
