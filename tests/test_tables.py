from convtools import conversion as c


def test_table_base_init():
    result = list(
        c.table.from_tuples(
            [(1, 2, 3), (2, 3, 4)], ["a", "b", "c"]
        ).into_iter_tuples(with_header=True)
    )
    assert result == [
        ("a", "b", "c"),
        (1, 2, 3),
        (2, 3, 4),
    ]
    result = list(
        c.table.from_tuples(
            [(1, 2, 3), (2, 3, 4)], ["a", "b", "c"]
        ).into_iter_dicts()
    )
    assert result == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 2, "b": 3, "c": 4},
    ]
