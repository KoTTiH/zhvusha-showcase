from pydantic import BaseModel


class ProjectCard(BaseModel):
    id: int
    title: str
    description: str
    price: int | None
    offers: int | None
    username: str
    url: str
    matched_keywords: list[str]


class DraftState(BaseModel):
    project_id: int
    project_title: str
    draft_text: str
