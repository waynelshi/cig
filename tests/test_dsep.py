from cig import PyCausalEngine


def test_chain_d_separation():
    # X -> M -> Y
    engine = PyCausalEngine()
    engine.add_extracted_edge("X", "M", None, None)
    engine.add_extracted_edge("M", "Y", None, None)

    # Marginally connected
    assert not engine.is_d_separated(["X"], ["Y"], [])
    # Conditioned on mediator M -> Blocked (d-separated)
    assert engine.is_d_separated(["X"], ["Y"], ["M"])


def test_fork_d_separation():
    # X <- C -> Y
    engine = PyCausalEngine()
    engine.add_extracted_edge("C", "X", None, None)
    engine.add_extracted_edge("C", "Y", None, None)

    # Marginally connected
    assert not engine.is_d_separated(["X"], ["Y"], [])
    # Conditioned on confounder C -> Blocked (d-separated)
    assert engine.is_d_separated(["X"], ["Y"], ["C"])


def test_collider_d_separation():
    # X -> W <- Y (Collider)
    engine = PyCausalEngine()
    engine.add_extracted_edge("X", "W", None, None)
    engine.add_extracted_edge("Y", "W", None, None)

    # Marginally blocked (d-separated naturally)
    assert engine.is_d_separated(["X"], ["Y"], [])
    # Conditioned on collider W -> Opens path (not d-separated)
    assert not engine.is_d_separated(["X"], ["Y"], ["W"])


def test_collider_descendant_activation():
    # X -> W <- Y
    #      |
    #      v
    #      D (Descendant)
    engine = PyCausalEngine()
    engine.add_extracted_edge("X", "W", None, None)
    engine.add_extracted_edge("Y", "W", None, None)
    engine.add_extracted_edge("W", "D", None, None)

    # Marginally blocked
    assert engine.is_d_separated(["X"], ["Y"], [])
    # Conditioned on descendant D -> Opens collider path through W!
    assert not engine.is_d_separated(["X"], ["Y"], ["D"])
