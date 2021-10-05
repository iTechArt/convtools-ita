import random
import statistics
from collections import Counter
from operator import eq
from typing import List, Tuple

import pytest

from convtools import conversion as c
from convtools.aggregations import MultiStatementReducer


class SumReducer1(MultiStatementReducer):
    reduce = ("%(result)s = {0} + ({1} or 1)",)
    default = int
    initial = int
    unconditional_init = True


class SumReducer2(MultiStatementReducer):
    reduce = ("%(result)s = {0} + ({1} or 2)",)
    default = 0
    initial = 0
    unconditional_init = True


class SumReducer3(MultiStatementReducer):
    reduce = ("%(result)s = {0} + ({1} or 3)",)
    initial = 0
    unconditional_init = True


class SumReducer4(MultiStatementReducer):
    prepare_first = ("%(result)s = {0}",)
    reduce = ("%(result)s = {0} + ({1} or 4)",)
    default = 0
    unconditional_init = True


class SumReducer5(MultiStatementReducer):
    reduce = ("%(result)s = {0} + ({1} or 5)",)
    default = 0
    unconditional_init = True


@pytest.fixture
def dict_series():
    return [
        {"name": "Nick", "value": 1},
        {"name": "Nick", "value": 2},
        {"name": "John", "value": 20},
        {"name": "John", "value": 21},
        {"name": "John", "value": 22},
    ]


@pytest.fixture
def series():
    random.seed(73)
    return [
        (random.randint(0, 100), random.randint(0, 10 ** 9))
        for _ in range(10000)
    ]


def ordered_set(data):
    return list({x: 1 for x in data})


def weighted_average(samples: List[Tuple]):
    return sum(v * w for v, w in samples) / sum(x[1] for x in samples)


def test_multi_statement_reducers(dict_series):
    output = (
        c.group_by(c.item("name"))
        .aggregate(
            (
                c.item("name"),
                SumReducer1(c.item("value")),
                SumReducer2(c.item("value")),
                SumReducer3(c.item("value")),
                SumReducer4(c.item("value")),
                SumReducer5(c.item("value"), initial=5),
            )
        )
        .execute(dict_series, debug=False)
    )
    assert output == [("Nick", 3, 3, 3, 3, 8), ("John", 63, 63, 63, 63, 68)]

    with pytest.raises(ValueError):

        class SumReducer(MultiStatementReducer):
            reduce = ("%(result)s = {0} + ({1} or 4)",)
            default = 0
            unconditional_init = True

        SumReducer(c.item("value"))
    with pytest.raises(ValueError):

        class SumReducer(MultiStatementReducer):
            reduce = ("%(result)s = {0} + ({1} or 4)",)
            unconditional_init = True

        SumReducer(c.item("value"))


def test_custom_reduce():
    with pytest.raises(ValueError):
        c.reduce(lambda a, b: a + b, c.this())
    with pytest.raises(ValueError):
        c.reduce(lambda a, b: a + b, c.this(), default=0)


def test_legacy_dict_reduce_approach(dict_series):
    output = c.aggregate(
        c.reduce(
            c.ReduceFuncs.DictSum,
            (c.item("name"), c.item("value")),
        )
    ).execute(dict_series)
    assert output == {
        "Nick": 3,
        "John": 63,
    }
    with pytest.raises(ValueError):
        c.ReduceFuncs.DictSum(c.this(), c.this(), c.this())
    with pytest.raises(ValueError):
        c.ReduceFuncs.DictSum({c.this(), c.this()})


def test_reducer_reuse(dict_series):
    f = lambda a, b: a + b
    reducer = c.reduce(f, c.item("value"), initial=0)
    reducer2 = c.reduce(f, c.item("value"), initial=0)
    output = (
        c.group_by(c.item("name"))
        .aggregate(
            (
                c.item("name"),
                reducer + 10,
                reducer2 + 20,
            )
        )
        .execute(dict_series)
    )
    assert output == [
        ("Nick", 13, 23),
        ("John", 73, 83),
    ]


def test_blank_aggregate(series):
    assert eq(
        c.group_by(c.item(0)).aggregate(c.item(0)).execute(series),
        list({x[0]: 1 for x in series}),
    )


def test_average(series):
    assert eq(
        c.aggregate(c.ReduceFuncs.Average(c.item(1))).execute(series, debug=True),
        statistics.mean(x[1] for x in series),
    )


def test_average_with_group_by(series):
    assert eq(
        c.group_by(c.item(0))
        .aggregate(c.ReduceFuncs.Average(c.item(1)))
        .execute(series),
        [
            statistics.mean(x[1] for x in series if x[0] == key)
            for key in ordered_set(x[0] for x in series)
        ],
    )


def test_average_of_empty_collection():
    assert c.aggregate(c.ReduceFuncs.Average(c.item(1))).execute([]) is None


def test_weighted_average(series):
    assert eq(
        c.aggregate(c.ReduceFuncs.Average(c.item(0), c.item(1))).execute(
            series
        ),
        weighted_average(series),
    )


def test_weighted_average_with_group_by(series):
    assert eq(
        c.group_by(c.item(0))
        .aggregate(c.ReduceFuncs.Average(c.item(0), c.item(1)))
        .execute(series),
        [
            weighted_average([x for x in series if x[0] == key])
            for key in ordered_set(x[0] for x in series)
        ],
    )


def test_mode(series):
    assert eq(
        c.aggregate(c.ReduceFuncs.Mode(c.item(0))).execute(series),
        statistics.mode(x[0] for x in series),
    )


def test_mode_with_groupby():
    series = [(0, 1), (0, 1), (0, 2), (1, 1), (1, 2), (1, 2)]

    assert eq(
        (
            c.group_by(c.item(0))
            .aggregate(c.ReduceFuncs.Mode(c.item(1)))
            .execute(series)
        ),
        [
            statistics.mode([x[1] for x in series if x[0] == key])
            for key in ordered_set(x[0] for x in series)
        ],
    )


@pytest.mark.parametrize("k", [1, 5, 10 ** 9])
def test_top_k(series, k):
    assert eq(
        c.aggregate(c.ReduceFuncs.TopK(k, c.item(1))).execute(series),
        [x[1] for x in Counter(x[1] for x in series).most_common(k)],
    )


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_non_positive_int(k):
    with pytest.raises(ValueError):
        c.aggregate(c.ReduceFuncs.TopK(k, c.this())).execute([1, 2]),


@pytest.mark.parametrize("k", [c.item(1), "abc"])
def test_top_k_invalid_input(k):
    with pytest.raises(TypeError):
        c.aggregate(c.ReduceFuncs.TopK(k, c.this())).execute([1, 2]),


@pytest.mark.parametrize("k", [1, 5])
def test_top_k_with_group_by(series, k):
    assert eq(
        c.group_by(c.item(0))
        .aggregate(c.ReduceFuncs.TopK(k, c.item(1)))
        .execute(series),
        [
            [
                x[1]
                for x in Counter(
                    x[1] for x in series if x[0] == key
                ).most_common(k)
            ]
            for key in ordered_set(x[0] for x in series)
        ],
    )


def test_median(series):
    assert eq(
        c.aggregate(c.ReduceFuncs.Median(c.item(1))).execute(series),
        statistics.median(x[1] for x in series),
    )


def test_median_with_group_by(series):
    assert eq(
        c.group_by(c.item(0))
        .aggregate(c.ReduceFuncs.Median(c.item(1)))
        .execute(series),
        [
            statistics.median(x[1] for x in series if x[0] == key)
            for key in ordered_set(x[0] for x in series)
        ],
    )


def test_multiple_aggregations(dict_series):
    assert (
        c.aggregate(c.ReduceFuncs.Array(c.item("name")))
        .pipe(
            c.aggregate(c.ReduceFuncs.ArrayDistinct(c.this())).pipe(
                c.aggregate(c.ReduceFuncs.Max(c.this()))
            )
        )
        .execute(dict_series, debug=False)
        == "Nick"
    )


def test_reducer_inlining(dict_series):
    def f():
        f.number_of_calls += 1
        if f.number_of_calls > f.max_number_of_calls:
            raise Exception
        return []

    f.max_number_of_calls = 1
    f.number_of_calls = 0

    converter = c.aggregate(
        c.ReduceFuncs.Array(
            c.item("name"), default=f, where=c.item("value") < 0
        ).pipe(
            c.if_(
                if_true=c.this(),
                if_false=c.this(),
            )
        )
    ).gen_converter(debug=False)
    assert converter(dict_series) == []


def test_group_by_key_edge_case():
    with pytest.raises(ValueError):
        c.this().add_label("row").pipe(c.ReduceFuncs.Count())
    with pytest.raises(ValueError):
        (c.this().add_label("row") + 1).pipe(c.ReduceFuncs.Count() + 1)
    with pytest.raises(ValueError):
        c.this().pipe(c.ReduceFuncs.Count(), label_input="row")
    data = [
        (0, 1),
        (1, 2),
    ]
    assert c.group_by(c.item(0)).aggregate(
        c.if_(c.item(1), c.item(1), c.item(1)).pipe(
            (c.ReduceFuncs.Sum(c.this()) / c.ReduceFuncs.Count(c.this())).pipe(
                c.this() + 10
            )
        )
    ).gen_converter(debug=False)(data) == [11, 12]
    assert c.group_by(c.item(0)).aggregate(
        c.item(1).pipe(c.ReduceFuncs.Sum(c.this()), label_output="count")
    ).gen_converter(debug=False)(data) == [1, 2]


def test_nested_group_by():
    data = [
        [0, [1, 2, 3]],
        [0, [4, 5, 6]],
        [1, [2, 3, 4]],
    ]
    assert c.group_by(c.item(0)).aggregate(
        (
            c.item(0),
            c.ReduceFuncs.Sum(
                c.item(1).pipe(c.aggregate(c.ReduceFuncs.Sum(c.this())))
            ),
        )
    ).execute(data, debug=False) == [
        (0, 21),
        (1, 9),
    ]
    agg_conv = c.aggregate(c.ReduceFuncs.Sum(c.this()))
    assert c.group_by(c.item(0)).aggregate(
        (
            c.item(0),
            c.if_(c.item(1), c.item(1), c.item(1),).pipe(
                c.if_(c.this(), c.this(), c.this(),).pipe(
                    c.ReduceFuncs.Sum(
                        c.if_(
                            c.this(),
                            c.this(),
                            c.this(),
                        )
                        .pipe((agg_conv, agg_conv))
                        .pipe(c.item(1))
                    ).pipe(
                        c.if_(
                            c.this(),
                            c.this(),
                            c.this(),
                        )
                    ),
                )
            ),
        )
    ).execute(data, debug=False) == [
        (0, 21),
        (1, 9),
    ]

    summer = c.aggregate(c.ReduceFuncs.Sum(c.this()))

    merger = c.aggregate(
        {
            "value1": c.ReduceFuncs.First(
                c.item("value1"), where=c("value1").in_(c.this())
            ),
            "value2": c.ReduceFuncs.First(
                c.item("value2"), where=c("value2").in_(c.this())
            ).pipe(c.if_(c.this(), c.this().pipe(summer))),
        }
    )
    converter = (
        c.group_by(c.item("id_"))
        .aggregate(
            {
                "id_": c.item("id_"),
                "data": c.ReduceFuncs.Array(c.this()).pipe(merger),
            }
        )
        .gen_converter(debug=False)
    )
    assert converter(
        [
            {"id_": 1, "value1": 2},
            {"id_": 2, "value1": 3},
            {"id_": 2, "value2": [1, 2, 3]},
        ]
    ) == [
        {"id_": 1, "data": {"value1": 2, "value2": None}},
        {"id_": 2, "data": {"value1": 3, "value2": 6}},
    ]

    def g():
        yield 1
        raise Exception

    assert (
        c.aggregate(c.ReduceFuncs.First(c.this())).execute(g(), debug=False)
    ) == 1


def test_aggregate_no_init_loops():
    converter = c.aggregate(
        {
            "first_a": c.ReduceFuncs.First(c.item("a"), where=c.item("b") > 0),
            "list_b": c.ReduceFuncs.Array(c.item("b"), where=c.item("a") > 0),
        }
    ).gen_converter(debug=False)
    assert converter(
        [
            {"a": 1, "b": 0},
            {"a": 2, "b": 1},
            {"a": 3, "b": 2},
            {"a": 4, "b": 3},
        ],
    ) == {
        "first_a": 2,
        "list_b": [0, 1, 2, 3],
    }
