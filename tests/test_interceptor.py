from cig import PyCausalEngine
from cig.hooks.interceptor import IDEInterceptor


def test_interceptor_detects_omitted_confounder():
    # DAG:
    # W -> T, W -> Y (Confounder W)
    # T -> Y
    engine = PyCausalEngine()
    engine.add_extracted_edge("w", "t", None, None)
    engine.add_extracted_edge("w", "y", None, None)
    engine.add_extracted_edge("t", "y", None, None)

    interceptor = IDEInterceptor(engine=engine)

    # Naive regression code written by an AI agent:
    code_snippet = """
    import statsmodels.formula.api as smf
    model = smf.ols(formula="y ~ t", data=df).fit()
    print(model.summary())
    """

    violations = interceptor.intercept_code(code_snippet)
    assert len(violations) == 1
    assert violations[0].violation_type == "OMITTED_CONFOUNDER"
    assert "w" in violations[0].problematic_variables
    assert violations[0].suggested_adjustment_set == ["w"]


def test_interceptor_passes_valid_backdoor_regression():
    engine = PyCausalEngine()
    engine.add_extracted_edge("w", "t", None, None)
    engine.add_extracted_edge("w", "y", None, None)
    engine.add_extracted_edge("t", "y", None, None)

    interceptor = IDEInterceptor(engine=engine)

    # Valid regression adjusting for confounder W:
    code_snippet = """
    import statsmodels.formula.api as smf
    model = smf.ols(formula="y ~ t + w", data=df).fit()
    """

    violations = interceptor.intercept_code(code_snippet)
    assert len(violations) == 0
