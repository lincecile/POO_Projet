from mypackage import LinearRegression

def test_simple():
    assert 1 + 1 == 2

def test_has_method():
    assert hasattr(LinearRegression, "fit")
    assert hasattr(LinearRegression, "predict")
    