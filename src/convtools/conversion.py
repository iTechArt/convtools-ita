"""
The main module exposing public API (via conversion object)
"""
import itertools

from .aggregations import Aggregate, BaseReducer, GroupBy, Reduce, ReduceFuncs
from .base import (
    And,
    BaseConversion,
    CallFunc,
    CodeGenerationOptionsCtx,
    ConversionException,
    ConverterOptionsCtx,
    Dict,
    DictComp,
    EscapedString,
    FilterConversion,
    GeneratorComp,
    GetAttr,
    GetItem,
    If,
    InlineExpr,
    InputArg,
    LabelConversion,
    List,
    ListComp,
    NaiveConversion,
    Not,
    OptionalCollectionItem,
    Or,
    Set,
    SetComp,
    SortConversion,
    This,
    Tuple,
    TupleComp,
    ensure_conversion,
)
from .joins import JoinConversion, _JoinConditions, ColumnRef
from .mutations import Mutations
from .tables import TableConversion


__all__ = ["conversion"]


class _Conversion:
    """The object, which exposes public API

    .. code-block:: python

      from convtools import conversion as c

      convert = c.aggregate(
          c.ReduceFuncs.DictSum(
              c.item("name"),
              c.item("value")
          )
      ).gen_converter(debug=True)

      assert convert([
          {"name": "Bob", "value": 10},
          {"name": "Bob", "value": 7},
          {"name": "Ron", "value": 3},
      ]) == {'Bob': 17, 'Ron': 3}

    """

    ConversionException = ConversionException  # pylint: disable=invalid-name
    BaseConversion = BaseConversion  # pylint: disable=invalid-name
    OptionsCtx = ConverterOptionsCtx  # pylint: disable=invalid-name
    CodeGenerationOptionsCtx = (  # pylint: disable=invalid-name
        CodeGenerationOptionsCtx
    )

    ReduceFuncs = ReduceFuncs  # pylint: disable=invalid-name
    #: Shortcut to `Mutations`
    Mut = Mutations  # pylint: disable=invalid-name

    and_ = And
    or_ = Or
    not_ = Not
    if_ = If

    this = This
    naive = NaiveConversion
    item = GetItem
    attr = GetAttr
    call_func = staticmethod(CallFunc)

    filter = FilterConversion
    sort = SortConversion

    #: Shortcut to `InputArg`
    input_arg = InputArg
    label = LabelConversion
    inline_expr = InlineExpr
    escaped_string = EscapedString

    #: Shortcut to ``List``
    list = List
    tuple = Tuple
    set = Set
    dict = Dict
    optional = OptionalCollectionItem

    generator_comp = GeneratorComp
    list_comp = ListComp
    tuple_comp = TupleComp
    set_comp = SetComp
    dict_comp = DictComp

    group_by = GroupBy
    aggregate = staticmethod(Aggregate)

    join = JoinConversion
    LEFT = _JoinConditions.LEFT
    RIGHT = _JoinConditions.RIGHT

    table = TableConversion
    col = ColumnRef

    def reduce(self, to_call_with_2_args, *args, **kwargs):
        if isinstance(to_call_with_2_args, type) and issubclass(
            to_call_with_2_args, BaseReducer
        ):
            return to_call_with_2_args(*args, **kwargs)
        return Reduce(to_call_with_2_args, *args, **kwargs)

    def __call__(self, obj: object):
        """Shortcut for ``ensure_conversion``"""
        return ensure_conversion(obj)

    def call(self, *args, **kwargs):
        return self.this().call(*args, **kwargs)

    def tap(self, *args, **kwargs):
        return self.this().tap(*args, **kwargs)

    def iter(self, *args, **kwargs):
        return self.this().iter(*args, **kwargs)

    def iter_mut(self, *args, **kwargs):
        return self.this().iter_mut(*args, **kwargs)

    def zip(self, *args, **kwargs):
        """Conversion which calls :py:obj:`zip` on conversions.

        Args:
          args: conversions to zip - returns tuples
          kwargs: named conversions to zip - returns dicts

        """
        if args and kwargs:
            raise ValueError("pass either args or kwargs")
        if args:
            return self.call_func(zip, *args)

        return self.call_func(zip, *kwargs.values()).iter(
            {name: self.item(index) for index, name in enumerate(kwargs)}
        )

    def repeat(self, obj, times=None):
        """shortcut to call :py:obj:`itertools.repeat`"""
        args = () if times is None else (times,)
        return self.call_func(itertools.repeat, obj, *args)

    def flatten(self):
        """c.this().flatten() shortcut"""
        return self.this().flatten()

    def min(self, arg1, arg2, *args):
        """c.call_func(min, ...) shortcut"""
        return self.call_func(min, arg1, arg2, *args)

    def max(self, arg1, arg2, *args):
        """c.call_func(max, ...) shortcut"""
        return self.call_func(max, arg1, arg2, *args)

    def breakpoint(self):
        """c.this().breakpoint() shortcut"""
        return self.this().breakpoint()


conversion = _Conversion()
