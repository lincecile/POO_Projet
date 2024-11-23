from mypackage import Strategy, Backtester, Result

def test_simple():
    assert 1 + 1 == 2

def test_has_method2():
    assert hasattr(Strategy, "fit")
    assert hasattr(Strategy, "get_position")

def test_has_method4():
    assert hasattr(Backtester, "run")

def test_has_method3():
    assert hasattr(Result, "_calculate_returns")
