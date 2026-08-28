from cig import PyCausalEngine


def test_backdoor_adjustment_search():
    # Confounder DAG:
    #   C -> X, C -> Y
    #   X -> M -> Y
    #   X -> Z_coll <- Y  (collider)
    engine = PyCausalEngine()
    engine.add_extracted_edge("C", "X", None, None)
    engine.add_extracted_edge("C", "Y", None, None)
    engine.add_extracted_edge("X", "M", None, None)
    engine.add_extracted_edge("M", "Y", None, None)
    engine.add_extracted_edge("X", "Z_coll", None, None)
    engine.add_extracted_edge("Y", "Z_coll", None, None)

    # Check manual adjustment sets
    assert not engine.is_valid_backdoor_set("X", "Y", [])
    assert engine.is_valid_backdoor_set("X", "Y", ["C"])
    assert not engine.is_valid_backdoor_set("X", "Y", ["M"])  # Descendant of X
    assert not engine.is_valid_backdoor_set(
        "X", "Y", ["Z_coll"]
    )  # Descendant of X & Collider

    # Automated search
    valid_sets = engine.find_backdoor_adjustment_sets("X", "Y")
    # Minimal valid set is ['C']
    assert ["C"] in valid_sets


def test_frontdoor_adjustment_search():
    # Hidden Confounder with Mediator:
    #   U -> X, U -> Y (Hidden confounder U)
    #   X -> M -> Y    (Mediator M)
    engine = PyCausalEngine()
    engine.add_extracted_edge("U", "X", None, None)
    engine.add_extracted_edge("U", "Y", None, None)
    engine.add_extracted_edge("X", "M", None, None)
    engine.add_extracted_edge("M", "Y", None, None)

    # Backdoor with empty or unobserved set fails
    assert not engine.is_valid_backdoor_set("X", "Y", [])

    # Frontdoor criterion on M
    assert engine.is_valid_frontdoor_set("X", "Y", ["M"])
    assert not engine.is_valid_frontdoor_set("X", "Y", ["U"])

    # Automated search
    frontdoor_sets = engine.find_frontdoor_adjustment_sets("X", "Y")
    assert ["M"] in frontdoor_sets
