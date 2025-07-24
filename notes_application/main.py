from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request


app = FastAPI()
models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")

class NotesBase(BaseModel):
    userid : int
    title : str
    description : str
    content : str
    


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session,Depends(get_db)]

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/notes/", status_code=status.HTTP_201_CREATED)
async def create_notes(notes:NotesBase, db:db_dependency):
    print("Received note:", notes)
    db_notes = models.Notes(**notes.dict())
    db.add(db_notes)
    db.commit()
    return {"message": "Note created"}

@app.get("/notes/{note_id}", status_code=status.HTTP_201_CREATED)
async def read_notes(note_id:int, db:db_dependency):
    note = db.query(models.Notes).filter(models.Notes.id == note_id).first()
    if note is None:
        HTTPException(status_code=404, detail = 'Note was not found')
    return note
    
@app.delete("/notes/{note_id}", status_code=status.HTTP_201_CREATED)
async def delete_notes(note_id:int, db:db_dependency):
    note = db.query(models.Notes).filter(models.Notes.id == note_id).first()
    if note is None:
        HTTPException(status_code=404, detail = 'Note was not found')
    db.delete(note)
    db.commit()







