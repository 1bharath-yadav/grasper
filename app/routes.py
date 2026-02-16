from fastapi import APIRouter, HTTPException, Request
from app.payload_handler import handle_payload

router = APIRouter()

# for official project to submit in my college


@router.post("/", tags=["Analysis"])
async def analyze_data(request: Request):
    form = await request.form()

    # Look for questions.txt either by form field name or by filename
    questions_file = None
    attachments = []

    for field_name, field_value in form.items():
        if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
            # Check if this is the questions file by field name or filename
            if field_name == "questions.txt" or field_value.filename == "questions.txt":  # type: ignore
                questions_file = field_value
            else:
                attachments.append(field_value)

    # Ensure questions.txt is present
    if not questions_file:
        raise HTTPException(
            status_code=400, detail="questions.txt file is required")

    # Read it
    try:
        # pyright: ignore[reportAttributeAccessIssue]
        input_text = (await questions_file.read()).decode("utf-8")

    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid questions.txt file encoding")

    payload = {
        "data_analyst_input": input_text,
        "attachments": attachments
    }

    return await handle_payload(payload)


# api to use promptfoo,streamlit and all other development things
# @router.post("/develop", tags=["PromptFoo"])
