from pydantic import BaseModel, Field


class PreferencePayload(BaseModel):
    gender: str = 'All'
    preferred_colors: list[str] = Field(default_factory=list)
    disliked_colors: list[str] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    preferred_types: list[str] = Field(default_factory=list)
    preferred_usage: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)


class InteractionPayload(BaseModel):
    item_id: int
    action: str
    source: str | None = 'ui'


class SearchPayload(BaseModel):
    user_id: str
    query: str
    top_k: int = 12


class RecommendPayload(BaseModel):
    user_id: str
    top_k: int = 12
