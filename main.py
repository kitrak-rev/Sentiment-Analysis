import uvicorn
import pyramid

from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
import pyramid_chameleon
import pyramid_swagger

from typing import Union

from fastapi import FastAPI,Form

from pydantic import BaseSettings
from fastapi.responses import HTMLResponse
from TrainedModel import BertClassifier

class Settings(BaseSettings):
    openapi_url: str = "/openapi.json"


settings = Settings()

app = FastAPI(openapi_url=settings.openapi_url)



@app.get("/")
def read_root():
    return "working"


@app.get("/predict/", response_class=HTMLResponse)
async def predict(text : str = None):
    if(text == None):
        return "<form action='/infer' method='post'> <input name='text' type='text'/> <input type='submit'/></form>"
    else:
        return "'" + text + "' is a " + BertClassifier().predict(text) + " Sentiment"

@app.post("/infer/",response_class=HTMLResponse)
async def infer(text : str = Form() ):
    return "<form action='.' method='post'> <input name='text'type='text'/> <input type='submit'/></form>"+"<br/> '"+text+"' is a "+BertClassifier().predict(text) + " Sentiment"

if __name__ == "__main__":
    pyramid.includes = pyramid_swagger
    config = Configurator(settings=settings)
    config.include('pyramid_chameleon')
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('api.things.get', '/predict', request_method='GET')
    config.add_route('api.things.post', '/', request_method='POST')
    
    # Additional routes go here
    
    config.scan()

    uvicorn.run("main:app",host='0.0.0.0', port=8000, reload=True, workers=3,)