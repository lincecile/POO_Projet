from mypackage import LinearRegression
from mypackage import Strategy

def test_simple():
    assert 1 + 1 == 2

def test_has_method():
    assert hasattr(LinearRegression, "fit")
    assert hasattr(LinearRegression, "predict")

def test_has_method2():
    assert hasattr(Strategy, "fit")
    assert hasattr(Strategy, "get_position")
