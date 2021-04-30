from pyramid.view import view_config

from celery_worker import app as celery_app, full_analyze_task #, init_worker_task

from celery.result import AsyncResult


@view_config(route_name='home', renderer='mls_analysis_service:templates/mytemplate.jinja2')
def my_view(request):
    print("aaaaa")
    return {}


@view_config(route_name='full_analyze', renderer='json', request_method='POST')
def full_analyze(request):
    # TODO: error handling when article_text or keywords is missing
    if(request.POST):
        article_text = request.POST['article']
        keywords = [s.strip() for s in request.POST['keyword'].split(',')]
        callback_url = request.POST['callbackUrl'] if 'callbackUrl' in request.POST else None
        print(keywords)
        # if callback_url is None:
        #     task = full_analyze_task.delay(input=article_text, language='ru', user_phrases=keywords)
        # else:
        task = full_analyze_task.apply_async(kwargs={'input':article_text, 'language':'ru', 'user_phrases':keywords, 'callback_url':callback_url}, ignore_result=(callback_url is None)) 
        return task.id if task is not None else None
    elif(request.json_body):
        print(request.json)
        article_text = request.json_body['article']
        keywords = request.json_body['keywords']
        callback_url = request.json_body['callbackUrl'] if 'callbackUrl' in request.json_body else None
        print(keywords)
        # if callback_url is None:
        #     task = full_analyze_task.delay(input=article_text, language='ru', user_phrases=keywords)
        # else:
        task = full_analyze_task.apply_async(kwargs={'input':article_text, 'language':'ru', 'user_phrases':keywords, 'callback_url':callback_url}, ignore_result=(callback_url is None)) 
        return task.id if task is not None else None

    return None


@view_config(route_name='check', renderer='json')
def check_task(request):
    res = AsyncResult(request.matchdict['task_id'], app=celery_app)
    print(request.matchdict['task_id'])
    if res.state == 'SUCCESS':
        return {
            'status': res.state,
            **res.result
        }
    else:
        return {
            'status': res.state
        }


@view_config(route_name='forget', renderer='json')
def forget_result(request):
    res = AsyncResult(request.matchdict['task_id'], app=celery_app)
    try:
        res.forget()
        return {"status": "success"}
    except:
        return {"status": "error"}

# @view_config(route_name='json', renderer='json', request_method='GET')
# def json_get(request):
#     # service = request.find_service(IExampleService)
#     # counter = service.example_function("string")
#     return {'project': 'mls_analysis_service'+counter}

@view_config(route_name='init_worker', renderer='json', request_method='POST')
def init_worker(request):
    task = full_analyze_task.delay()

@view_config(route_name='json', renderer='json', request_method='POST')
def json_post(request):
    return {
        "coherence": [
            {
                "has_missing": 0,
                "missing_sentences": [
                    0,
                    0,
                    0
                ],
                "has_incoherent": 0,
                "incoherent_sentences": [
                    0,
                    0,
                    0,
                    0
                ]
            },
            {}
        ],
        "user_phrases": [
            {
                "text": "логструктурированный кэш",
                "score": 0.5866227746009827
            },
            {
                "text": "параллельный кэш",
                "score": 0.0021245565731078386
            }
        ],
        "keyword_candidates": [
            {
                "text": "ssd",
                "score": 0.8878007531166077
            },
            {
                "text": "hit",
                "score": 0.8843870759010315
            }
        ]
    }
