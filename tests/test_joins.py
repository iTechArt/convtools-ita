import pytest

from convtools import conversion as c
from convtools.base import Eq
from convtools.conversion import _JoinConditions


def test_join_conditions():
    join_conditions = _JoinConditions.from_condition(c.LEFT == c.RIGHT)
    assert (
        True
        and join_conditions.inner_loop_conditions == []
        and join_conditions.left_collection_filters == []
        and join_conditions.left_row_filters == []
        and join_conditions.left_row_hashers == [c.LEFT]
        and join_conditions.pre_filter == []
        and join_conditions.right_collection_filters == []
        and join_conditions.right_row_filters == []
        and join_conditions.right_row_hashers == [c.RIGHT]
    )

    join_conditions = _JoinConditions.from_condition(
        c.or_(c.LEFT == c.RIGHT, c.LEFT == c.RIGHT)
    )

    c11 = c.LEFT.item(0)
    c12 = c.RIGHT.item(1)
    c21 = c.LEFT.item(1)
    c22 = c.RIGHT.item(0)
    c13 = c.LEFT.item(2) > 10
    c23 = c.RIGHT.item(2) < 10
    c01 = c.input_arg("x") > 100
    join_conditions = _JoinConditions.from_condition(
        c.and_(c11 == c12, c22 == c21, c13).and_(c23, c01)
    )
    assert (
        True
        and join_conditions.inner_loop_conditions == []
        and join_conditions.left_collection_filters == [c13]
        and join_conditions.left_row_filters == []
        and join_conditions.left_row_hashers == [c11, c21]
        and join_conditions.pre_filter == [c01]
        and join_conditions.right_collection_filters == [c23]
        and join_conditions.right_row_filters == []
        and join_conditions.right_row_hashers == [c12, c22]
    )
    join_conditions = _JoinConditions.from_condition(
        c.and_(c11 == c12, c22 == c21, c13).and_(c23, c01), how="left"
    )
    assert (
        True
        and join_conditions.inner_loop_conditions == []
        and join_conditions.left_collection_filters == []
        and join_conditions.left_row_filters == [c13]
        and join_conditions.left_row_hashers == [c11, c21]
        and join_conditions.pre_filter == [c01]
        and join_conditions.right_collection_filters == [c23]
        and join_conditions.right_row_filters == []
        and join_conditions.right_row_hashers == [c12, c22]
    )
    join_conditions = _JoinConditions.from_condition(
        c.and_(c11 == c12, c22 == c21, c13).and_(c23, c01), how="right"
    )
    assert (
        True
        and join_conditions.inner_loop_conditions == []
        and join_conditions.left_collection_filters == [c13]
        and join_conditions.left_row_filters == []
        and join_conditions.left_row_hashers == [c11, c21]
        and join_conditions.pre_filter == [c01]
        and join_conditions.right_collection_filters == []
        and join_conditions.right_row_filters == [c23]
        and join_conditions.right_row_hashers == [c12, c22]
    )
    join_conditions = _JoinConditions.from_condition(
        c.and_(c11 == c12, c22 == c21, c13).and_(c23, c01), how="outer"
    )
    assert (
        True
        and join_conditions.inner_loop_conditions == []
        and join_conditions.left_collection_filters == []
        and join_conditions.left_row_filters == [c13]
        and join_conditions.left_row_hashers == [c11, c21]
        and join_conditions.pre_filter == [c01]
        and join_conditions.right_collection_filters == []
        and join_conditions.right_row_filters == [c23]
        and join_conditions.right_row_hashers == [c12, c22]
    )
    with pytest.raises(AssertionError):
        _JoinConditions.from_condition(c.and_(True, False), how="abc")

    c1 = c.LEFT != c.RIGHT
    join_conditions = _JoinConditions.from_condition(c1)
    assert (
        True
        and join_conditions.inner_loop_conditions == [c1]
        and join_conditions.left_collection_filters == []
        and join_conditions.left_row_filters == []
        and join_conditions.left_row_hashers == []
        and join_conditions.pre_filter == []
        and join_conditions.right_collection_filters == []
        and join_conditions.right_row_filters == []
        and join_conditions.right_row_hashers == []
    )

    cond = c.LEFT > c.RIGHT
    join_conditions = _JoinConditions.from_condition(cond)
    assert (
        True
        and join_conditions.inner_loop_conditions == [cond]
        and join_conditions.left_collection_filters == []
        and join_conditions.left_row_filters == []
        and join_conditions.left_row_hashers == []
        and join_conditions.pre_filter == []
        and join_conditions.right_collection_filters == []
        and join_conditions.right_row_filters == []
        and join_conditions.right_row_hashers == []
    )

    c1 = c.LEFT == 1
    c2 = c.RIGHT == 1
    c3 = c.input_arg("x") == 1
    join_conditions = _JoinConditions.from_condition(
        c1.and_(c2, c3), how="outer"
    )
    assert (
        True
        and join_conditions.inner_loop_conditions == []
        and join_conditions.left_collection_filters == []
        and join_conditions.left_row_filters == [c1]
        and join_conditions.left_row_hashers == []
        and join_conditions.pre_filter == [c3]
        and join_conditions.right_collection_filters == []
        and join_conditions.right_row_filters == [c2]
        and join_conditions.right_row_hashers == []
    )

    c1 = c.LEFT + c.RIGHT + 10
    c2 = c.LEFT + 1
    c3 = c.RIGHT + 1
    join_conditions = _JoinConditions.from_condition(c.and_(c1, c2, c3))
    assert (
        True
        and join_conditions.inner_loop_conditions == [c1]
        and join_conditions.left_collection_filters == [c2]
        and join_conditions.left_row_filters == []
        and join_conditions.left_row_hashers == []
        and join_conditions.pre_filter == []
        and join_conditions.right_collection_filters == [c3]
        and join_conditions.right_row_filters == []
        and join_conditions.right_row_hashers == []
    )


def test_hash_joins():

    join1 = (
        c.join(c.item(0), c.item(1), c.LEFT.item("id") == c.RIGHT.item("id"))
        .as_type(list)
        .gen_converter(debug=False)
    )
    join1(
        [
            [{"id": i, "value": i + 100} for i in range(3)],
            [{"id": i, "value": i + 200} for i in range(3)],
        ]
    ) == [
        ({"id": 0, "value": 100}, {"id": 0, "value": 200}),
        ({"id": 1, "value": 101}, {"id": 1, "value": 201}),
        ({"id": 2, "value": 102}, {"id": 2, "value": 202}),
    ]

    join2 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(
                c.LEFT.item("id") == c.RIGHT.item("ID"),
                c.LEFT.item("value") > 105,
                c.RIGHT.item("value") < 209,
                c.input_arg("flag"),
            ),
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join2(
        [
            [{"id": i, "value": i + 100} for i in range(10)],
            [{"ID": i, "value": i + 200} for i in range(10)],
        ],
        flag=True,
    ) == [
        ({"id": 6, "value": 106}, {"ID": 6, "value": 206}),
        ({"id": 7, "value": 107}, {"ID": 7, "value": 207}),
        ({"id": 8, "value": 108}, {"ID": 8, "value": 208}),
    ]
    assert (
        join2(
            [
                [{"id": i, "value": i + 100} for i in range(10)],
                [{"ID": i, "value": i + 200} for i in range(10)],
            ],
            flag=False,
        )
        == []
    )

    join2 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(
                c.LEFT.item("id") == c.RIGHT.item("ID"),
                c.LEFT.item("value") > 105,
                c.RIGHT.item("value") < 209,
                c.input_arg("flag"),
            ),
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join2(
        [
            [{"id": i, "value": i + 100} for i in range(10)],
            [{"ID": i, "value": i + 200} for i in range(10)],
        ],
        flag=True,
    ) == [
        ({"id": 6, "value": 106}, {"ID": 6, "value": 206}),
        ({"id": 7, "value": 107}, {"ID": 7, "value": 207}),
        ({"id": 8, "value": 108}, {"ID": 8, "value": 208}),
    ]


def test_nested_loop_joins():
    join1 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(
                c.LEFT.item("id") - 1 == c.RIGHT.item("ID"),
                c.LEFT.item("value") + 100 < c.RIGHT.item("value"),
                c.LEFT.item("value") > 101,
                c.RIGHT.item("value") < 209,
                c.input_arg("flag"),
            ),
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join1(
        [
            [{"id": i, "value": i + 100} for i in range(10)],
            [{"ID": i, "value": 210 - i} for i in range(10)],
        ],
        flag=True,
    ) == [
        ({"id": 3, "value": 103}, {"ID": 2, "value": 208}),
        ({"id": 4, "value": 104}, {"ID": 3, "value": 207}),
        ({"id": 5, "value": 105}, {"ID": 4, "value": 206}),
    ]

    join2 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(c.LEFT * c.RIGHT < 30, c.LEFT + c.RIGHT > 8),
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join2([range(4, 10), range(4, 10)]) == [
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 4),
        (5, 5),
        (6, 4),
        (7, 4),
    ]
    join3 = (
        c.join(c.item(0), c.item(1), Eq(c.LEFT + c.RIGHT, 1))
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join3(([-1, 0, 1], [2, 1, 1])) == [(-1, 2), (0, 1), (0, 1)]


def test_left_join():
    join1 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(
                c.LEFT == c.RIGHT,
                c.LEFT + c.RIGHT < 10,
                c.LEFT > 0,
            ),
            how="left",
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join1([(0, 1, 2, 3, 3), (3, 3, 4, 5)]) == [
        (0, None),
        (1, None),
        (2, None),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]

    conv = (
        c.join(
            c.item("first"),
            c.item("second"),
            (
                c.LEFT.item("name").call_method("lower")
                == c.RIGHT.item("full_name").call_method("lower")
            ),
            how="left",
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    # fmt: off
    assert conv(
        {
            "first": [
                {"name": "JOHN"},
                {"name": "JOHN"},
                {"name": "bob"},
                {"name": "bob"},
                {"name": "ron"},
                {"name": "ron"},
            ],
            "second": [
                {"full_name": "BOB"},
                {"full_name": "John"},
                {"full_name": "Nick"},
            ],
        }
    ) == [
        ({"name": "JOHN"}, {"full_name": "John"},),
        ({"name": "JOHN"}, {"full_name": "John"},),
        ({"name": "bob"}, {"full_name": "BOB"},),
        ({"name": "bob"}, {"full_name": "BOB"},),
        ({"name": "ron"}, None),
        ({"name": "ron"}, None),
    ]
    # fmt: on


def test_right_join():
    join1 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(
                c.LEFT == c.RIGHT,
                c.LEFT + c.RIGHT < 10,
                c.LEFT > 0,
            ),
            how="right",
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join1([(0, 1, 2, 3, 3), (3, 3, 4, 5)]) == [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (None, 4),
        (None, 5),
    ]

    conv = (
        c.join(
            c.item("first"),
            c.item("second"),
            (
                c.LEFT.item("name").call_method("lower")
                == c.RIGHT.item("full_name").call_method("lower")
            ),
            how="right",
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    # fmt: off
    assert conv(
        {
            "first": [
                {"name": "JOHN"},
                {"name": "bob"},
                {"name": "ron"},
            ],
            "second": [
                {"full_name": "BOB"},
                {"full_name": "BOB"},
                {"full_name": "John"},
                {"full_name": "Nick"},
                {"full_name": "Nick"},
            ],
        }
    ) == [
        ({"name": "bob"}, {"full_name": "BOB"},),
        ({"name": "bob"}, {"full_name": "BOB"},),
        ({"name": "JOHN"}, {"full_name": "John"},),
        (None, {"full_name": "Nick"}),
        (None, {"full_name": "Nick"}),
    ]
    # fmt: on


def test_outer_join():
    join1 = (
        c.join(
            c.item(0),
            c.item(1),
            Eq(c.LEFT, c.RIGHT, 2),
            how="full",
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join1(([0, 1, 2, 5], [2, 3, 4, 5])) == [
        (0, None),
        (1, None),
        (2, 2),
        (5, None),
        (None, 3),
        (None, 4),
        (None, 5),
    ]
    join2 = (
        c.join(
            c.item(0),
            c.item(1),
            c.and_(
                c.LEFT == c.RIGHT,
                c.LEFT + c.RIGHT < 10,
                c.LEFT > 0,
            ),
            how="outer",
        )
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join2([(10, 7, 8, 0, 1, 2, 3, 3), (3, 3, 4, 5, 8)]) == [
        (10, None),
        (7, None),
        (8, None),
        (0, None),
        (1, None),
        (2, None),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (None, 4),
        (None, 5),
        (None, 8),
    ]


def test_cross_join():
    join1 = (
        c.join(c.item(0), c.item(1), True)
        .as_type(list)
        .gen_converter(debug=False)
    )
    assert join1(([1, 2, 3], [5, 6])) == [
        (1, 5),
        (1, 6),
        (2, 5),
        (2, 6),
        (3, 5),
        (3, 6),
    ]


def test_join_bad_inputs():
    with pytest.raises(ValueError):
        c.join(c.item(0), c.item(1), True, "hz")


def test_join_with_input_args():
    assert (
        c.join(
            c.input_arg("custom_left"),
            c.input_arg("custom_right"),
            c.LEFT == c.RIGHT,
        )
        .as_type(list)
        .execute(None, custom_left=range(3), custom_right=range(3), debug=True)
    ) == [(0, 0), (1, 1), (2, 2)]


def test_join_with_complex_pipe():
    def f(l):
        return l + [1, 3]

    pipeline = (
        c.aggregate(c.ReduceFuncs.Array(c.item("a")))
        .pipe(c.join(c.this(), c.call_func(f, c.this()), c.LEFT == c.RIGHT))
        .iter(c.item(1))
        .as_type(list)
    )

    assert (
        pipeline.execute(
            [
                {"a": 1},
                {"a": 2},
                {"a": 3},
            ]
        )
        == [1, 1, 2, 3, 3]
    )


"""
abc_rows = c.Rows.from_csv("abc.csv", with_header=True)
abc = abc_rows.as_columns()

# row-wise reading / col-wise reading


(
    c.Table.from_csv("apps.csv")
    .join(c.Table.from_csv("subs.csv"), on=["company_id"])
    .update(total_revenue=c.col("app_revenue") + c.col("sub_revenue"))
    .to_csv("results.csv")
)


columns.update()  # {}
columns.take()  # []
columns.drop()  # []

columns.join()
columns.concat()

columns.raw()
columns.iter_tuples()
columns.iter_dicts()

result = (
    abc.update(
        {
            "app_id": abc.col("Application ID").as_type(int),
            "app_name": abc.col("Application Name").call_method("strip"),
            "proceeds": abc.col("Proceeds").as_type(Decimal),
        }
    )
    .take(["app_id", "app_name", "proceeds"])
    .join(
        cde.update(
            {
                "sales": cde.col("Sales").as_type(Decimal),
            }
        ).drop(["category"]),
        # on=c.LEFT.col("app_id") == c.RIGHT.col("app_id"),
        on=["app_id"],
        how="left",
    )
    .join(
        efg,
        on=["app_id"],
        how="left",
        suffixes=("", "_RIGHT"),
    )
)
result.update(
    {"fee_percentage": result.col("proceeds") / result.col("sales") * -1 + 1}
).take(["company_name", "app_id", "app_name", "fee_percentage"]).to_csv(
    "result.csv"
)
abc.concat(cde).to_csv("result2.csv")
"""
