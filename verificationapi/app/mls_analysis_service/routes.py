def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('full_analyze', '/full_analyze')
    config.add_route('json', '/json')
    config.add_route('check', '/check/{task_id}')
    config.add_route('forget', '/forget/{task_id}')
    config.add_route('init_worker', '/init_worker')
        