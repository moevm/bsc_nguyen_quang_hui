from pyramid.view import view_config

from celery_worker import app as celery_app, full_analyze_task

from celery.result import AsyncResult


@view_config(route_name='home', renderer='mls_analysis_service:templates/mytemplate.jinja2')
def my_view(request):
    print("aaaaa")
    return {}


@view_config(route_name='full_analyze', renderer='json', request_method='POST')
def full_analyze(request):
    if(request.POST):
        article_text = request.POST['article']
        keywords = [s.strip() for s in request.POST['keyword'].split(',')]
        print(keywords)
        task = full_analyze_task.delay(article_text, 'ru', user_phrases=keywords)
        return task.id
    elif(request.json_body):
        print(request.json)
        article_text = request.json_body['article']
        keywords = request.json_body['keywords']
        print(keywords)
        task = full_analyze_task.delay(article_text, 'ru', user_phrases=keywords)
        return task.id

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
