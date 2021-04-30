import os 
import time
from celery import Celery
import celery
from analizers import ArticleAnalizingService
import requests
        
print("celery queue init function")
# app = initCeleryApp()
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_BACKEND', 'redis://localhost:6379')

class Config:
    mongodb_backend_settings = {
        'database':"verification_api",
        'taskmeta_collection':"results"
    }

app = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND, config_source=Config)


class AnalysisTask(app.Task):
    _service = None
    
    @property
    def service(self):
        if self._service is None:
            print("init db")
            self._service = ArticleAnalizingService()
        return self._service

    def __init__(self):
        # This is called once when registering Task/using this task class as base for task, 
        # so we can't initiate Service in here, else it will be called on both containers
        # print("task init")
        # self.ignore_result = False
        pass
        
    def run(self, x, y):
        # can be called with super() in inherited tasks
        # print('running task add')
        # print(x+y)
        # return x+y
        pass
        
@app.task(bind=True, base=AnalysisTask)
def full_analyze_task(self, input=None, language='en', user_phrases=[], extractor_n = 10, callback_url = None):
    print("Input received: "+str(input is None) + "; callbackUrl: "+str(callback_url))
    # if input is not available, just load the model then finish
    if input is None:
        self.service
        return None

    # if callback_url is available, process it then send callback
    if callback_url is not None:
        # process the article
        result = self.service.full_analyze(input, language, user_phrases=user_phrases, extractor_n = extractor_n)
        # send a post request for webhook response
        requests.post(callback_url, json= result)
        # pass
        return None
    else:
        # if input is available, process it then save result
        return self.service.full_analyze(input, language, user_phrases=user_phrases, extractor_n = extractor_n)
    
# NOTE: this create task from base task, effectively - another task, which load its own model
# @app.task(bind=True, base=AnalysisTask, ignore_result=True)
# def init_worker_task(self):
#     self.service