from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.core.translate import translate_text  

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

class TranslateRequest(BaseModel):
    text: str

@router.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "translation": None})

@router.post("/translate")
async def post_translate(data: TranslateRequest):
    translation = translate_text(data.text)
    return JSONResponse(content={"translation": translation})
