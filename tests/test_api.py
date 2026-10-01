def test_api_routes():
    from src.api.main import app
    paths={r.path for r in app.routes} 
    assert {"/health","/analyze","/chat","/retrieve","/predict-intent"}<=paths
