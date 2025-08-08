from pydantic import BaseModel
from typing import Optional, List


class AnalysisRequest(BaseModel):
    data_analyst_input: str
    attachments: Optional[List[str]] = None
