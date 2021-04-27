import os 
import time
from celery import Celery
import celery
from analizers import ArticleAnalizingService

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
        
@app.task(bind=True, base=AnalysisTask, ignore_result=False)
def full_analyze_task(self, input, language, user_phrases=[], extractor_n = 10):
    return self.service.full_analyze(input, language, user_phrases=user_phrases, extractor_n = extractor_n)
    
# @app.task(bind=True, base=AnalysisTask)
# def custom(self,x,y):
#     print(x+y)
#     self.db
#     self.db
#     self.db
#     self.db
#     print('running custom')

# simple task
# @app.task
# def add(x, y):
#     print("called")
#     time.sleep(5)
#     print("wait completed")
#     return x + y
