from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import models
from database import SessionLocal, engine
from typing import Annotated

app = FastAPI()
models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")


class NotesBase(BaseModel):
    title: str
    description: str
    content: str
    


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
async def get_search(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/delete", response_class=HTMLResponse)
async def get_delete(request: Request):
    return templates.TemplateResponse("delete.html", {"request": request})

@app.post("/notes/", status_code=status.HTTP_201_CREATED)
async def create_note(note: NotesBase, db: db_dependency):
    db_note = models.Notes(**note.dict())
    db.add(db_note)
    db.commit()
    return {"message": "Note created successfully"}

@app.get("/notes/{note_id}")
async def get_note(note_id: int, db: db_dependency):
    note = db.query(models.Notes).filter(models.Notes.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

@app.delete("/notes/{note_id}")
async def delete_note(note_id: int, db: db_dependency):
    note = db.query(models.Notes).filter(models.Notes.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(note)
    db.commit()
    return {"message": "Note deleted successfully"}
