"""This module brings aggregations with various reduce functions"""
import statistics
import typing
from collections import defaultdict

from .base import (
    BaseConversion,
    CallFunc,
    CodeGenerationOptionsCtx,
    ConversionException,
    EscapedString,
    FilterConversion,
    GetItem,
    InlineExpr,
    List,
    NaiveConversion,
    Tuple,
    _None,
)
from .utils import Code


_none = BaseConversion._none
RBT = typing.TypeVar("RBT", bound="ReduceBlock")


class ReduceBlock:
    """Represents a section of code of a single reducer"""

    var_checksum = "checksum_"
    var_expected_checksum = "expected_checksum_"

    def __init__(
        self,
        reduce_initial: typing.Iterable[str],
        reduce_two: typing.Iterable[str],
        var_row: str,
        var_agg_data_value: str,
        checksum_flag: int,
        unconditional_init: bool,
    ):
        self.reduce_initial = {var_agg_data_value: reduce_initial}
        self.reduce_two = {var_agg_data_value: reduce_two}
        self.var_row = var_row
        self.condition_code = None
        self.unconditional_init = unconditional_init
        self.checksum_flag = checksum_flag

    def consume(self: RBT, reduce_block: RBT) -> RBT:
        if (
            self.condition_code != reduce_block.condition_code
            or self.unconditional_init != reduce_block.unconditional_init
            or self.var_row != reduce_block.var_row
        ):
            raise AssertionError

        for k, v in reduce_block.reduce_initial.items():
            if k in self.reduce_initial:
                raise AssertionError
            self.reduce_initial[k] = v

        for k, v in reduce_block.reduce_two.items():
            if k in self.reduce_two:
                raise AssertionError
            self.reduce_two[k] = v

        self.checksum_flag |= reduce_block.checksum_flag
        return self

    def iter_reduce_lines(self, lines_info_dict) -> typing.Iterable[str]:
        for var_agg_data_value, lines in lines_info_dict.items():
            template_kwargs = {
                "result": var_agg_data_value,
                "row": self.var_row,
            }
            yield from (line % template_kwargs for line in lines)

    def iter_reduce_initial_lines(self) -> typing.Iterable[str]:
        yield from self.iter_reduce_lines(self.reduce_initial)

    def iter_reduce_two_lines(self) -> typing.Iterable[str]:
        yield from self.iter_reduce_lines(self.reduce_two)

    def same_as(self, other):
        return (
            self.condition_code == other.condition_code
            and self.unconditional_init == other.unconditional_init
            and self.var_row == other.var_row
            and len(self.reduce_initial) == len(other.reduce_initial)
            and len(self.reduce_two) == len(other.reduce_two)
            and all(
                l1 == l2
                for l1, l2 in zip(
                    self.reduce_initial.values(), other.reduce_initial.values()
                )
            )
            and all(
                l1 == l2
                for l1, l2 in zip(
                    self.reduce_two.values(), other.reduce_two.values()
                )
            )
        )


class ReduceConditionalBlock(ReduceBlock):
    """Represents a section of code of a single reducer with an incoming
    condition"""

    def __init__(self, *args, **kwargs):
        condition_code = kwargs.pop("condition_code")
        super().__init__(*args, **kwargs)
        self.condition_code = condition_code


class ReduceBlocks(typing.Generic[RBT]):
    """Represents a set of reduce blocks"""

    def __init__(self):
        self.conditional_init_blocks = {}
        self.unconditional_init_blocks = {}
        self.number = 0

    def add_block(self, reduce_block: RBT):
        self.number += 1
        if reduce_block.unconditional_init:
            key_to_block = self.unconditional_init_blocks
        else:
            key_to_block = self.conditional_init_blocks
        key = (
            (
                reduce_block.condition_code
                if isinstance(reduce_block, ReduceConditionalBlock)
                else None
            ),
            reduce_block.var_row,
        )
        if key in key_to_block:
            key_to_block[key].consume(reduce_block)
        else:
            key_to_block[key] = reduce_block

    def iter_blocks(self) -> typing.Iterable[RBT]:
        yield from self.unconditional_init_blocks.values()
        yield from self.conditional_init_blocks.values()

    def gen_group_by_code(
        self, var_row, var_agg_data, var_signature_to_agg_data, code_signature
    ) -> Code:
        code = Code()
        code.add_line(f"for {var_row} in data_:", 1)
        code.add_line(
            f"{var_agg_data} = {var_signature_to_agg_data}[{code_signature}]",
            0,
        )

        for block in self.iter_blocks():
            reduce_initial_lines = list(block.iter_reduce_initial_lines())
            reduce_two_lines = list(block.iter_reduce_two_lines())
            any_var_agg_data_value = next(iter(block.reduce_initial))

            if block.condition_code:
                code.add_line(f"if {block.condition_code}:", 1)

            code.add_line(f"if {any_var_agg_data_value} is _none:", 1)
            for line in reduce_initial_lines:
                code.add_line(line, 0)
            code.incr_indent_level(-1)

            if reduce_two_lines:
                code.add_line("else:", 1)
                for line in reduce_two_lines:
                    code.add_line(line, 0)
                code.incr_indent_level(-1)

            if block.condition_code:
                code.incr_indent_level(-1)
        return code

    def gen_aggregate_code(
        self, var_row, var_checksum, var_expected_checksum
    ) -> Code:
        first_phase_code = Code()
        second_phase_code = Code()

        needs_sum_checking = False

        reduce_blocks = list(self.iter_blocks())
        is_single_block = len(reduce_blocks) == 1
        for block in reduce_blocks:
            reduce_initial_lines = list(block.iter_reduce_initial_lines())
            reduce_two_lines = list(block.iter_reduce_two_lines())
            any_var_agg_data_value = next(iter(block.reduce_initial))

            init_phase_needs_else = not (
                is_single_block and block.unconditional_init
            )

            if block.condition_code:
                first_phase_code.add_line(f"if {block.condition_code}:", 1)

            first_phase_code.add_line(
                f"if {any_var_agg_data_value} is _none:", 1
            )
            for line in reduce_initial_lines:
                first_phase_code.add_line(line, 0)

            if is_single_block:
                if block.unconditional_init:
                    first_phase_code.add_line("break", 0)
                else:
                    first_phase_code.add_line(
                        f"if {any_var_agg_data_value} is not _none:", 1
                    )
                    first_phase_code.add_line("break", -1)

            else:
                if block.unconditional_init:
                    first_phase_code.add_line(
                        f"{var_checksum} |= {block.checksum_flag}", 0
                    )
                else:
                    first_phase_code.add_line(
                        f"if {any_var_agg_data_value} is not _none:", 1
                    )
                    first_phase_code.add_line(
                        f"{var_checksum} |= {block.checksum_flag}", -1
                    )
                needs_sum_checking = True

            first_phase_code.incr_indent_level(-1)

            if init_phase_needs_else and reduce_two_lines:
                first_phase_code.add_line("else:", 1)
                for line in reduce_two_lines:
                    first_phase_code.add_line(line, 0)
                first_phase_code.incr_indent_level(-1)

            if block.condition_code:
                first_phase_code.incr_indent_level(-1)

            if reduce_two_lines:
                if block.condition_code:
                    second_phase_code.add_line(
                        f"if {block.condition_code}:", 1
                    )

                for line in reduce_two_lines:
                    second_phase_code.add_line(line, 0)

                if block.condition_code:
                    second_phase_code.incr_indent_level(-1)

        resulting_code = Code()
        resulting_code.add_line("it_ = iter(data_)", 0)
        resulting_code.add_line(f"for {var_row} in it_:", 1)
        resulting_code.add_code(first_phase_code)
        if needs_sum_checking:
            resulting_code.add_line(
                f"if {var_checksum} == {var_expected_checksum}: break", 0
            )
        resulting_code.incr_indent_level(-1)

        if second_phase_code.has_lines():
            resulting_code.add_line("", 0)
            resulting_code.add_line(f"for {var_row} in it_:", 1)
            resulting_code.add_code(second_phase_code)
            resulting_code.incr_indent_level(-1)
        return resulting_code


GROUPER_TEMPLATE = """
def {converter_name}(data_{code_args}):
    {var_signature_to_agg_data} = defaultdict({var_agg_data_cls})

{code_group_by}

{code_result}
"""
AGGREGATE_TEMPLATE = """
def {converter_name}(data_{code_args}):
    {code_init_agg_vars}
    {var_expected_checksum} = {val_expected_checksum}
    {var_checksum} = 0

{code_aggregate}

{code_result}
"""


RT = typing.TypeVar("RT", bound="BaseReducer")


class BaseReducer(BaseConversion, typing.Generic[RBT]):
    """Base of a reduce operation to be used during the aggregation"""

    method_calls_replace_input_with_self = True

    expressions: typing.Tuple[typing.Any, ...]
    post_conversion: typing.Optional[BaseConversion] = None
    default: typing.Any
    initial: typing.Any
    where: typing.Optional[BaseConversion] = None
    unconditional_init: bool = False

    self_content_type = BaseConversion.ContentTypes.REDUCER

    def gen_reduce_code_block(
        self,
        var_agg_data_value: str,
        var_row: str,
        checksum_flag: int,
        ctx: dict,
    ) -> RBT:
        raise NotImplementedError

    def _gen_code_and_update_ctx(self, code_input, ctx) -> str:
        reducers_run_stage = CodeGenerationOptionsCtx.get_option_value(
            "reducers_run_stage"
        )
        if reducers_run_stage == "collecting_reducer_inputs":
            reducer_inputs_info = ctx["_reducer_inputs_info"][-1]
            if reducer_inputs_info is not False:
                reducer_inputs_info[code_input].append(self)
            agg_data_item = code_input
        elif reducers_run_stage == "rendering_reducer_results":
            overwritten_reducer_inputs = ctx["_overwritten_reducer_inputs"][-1]
            agg_data_item = overwritten_reducer_inputs[(code_input, self)]
        else:
            raise AssertionError(
                "reducers cannot be used outside of aggregations"
            )

        processed_agg_data_item = agg_data_item
        if self.post_conversion:
            processed_agg_data_item = (
                self.post_conversion.gen_code_and_update_ctx(
                    agg_data_item, ctx
                )
            )
        if hasattr(self, "default"):
            default_value = self.default
        elif hasattr(self, "initial"):
            default_value = self.initial
        else:
            raise AssertionError

        default_value_code = default_value.gen_code_and_update_ctx(None, ctx)
        return EscapedString(
            f"({default_value_code} "
            f"if {agg_data_item} is _none "
            f"else {processed_agg_data_item})"
        ).gen_code_and_update_ctx("", ctx)


class MultiStatementReducer(BaseReducer):
    """Defines the reduce operation, which is based on multiple python
    statements, to be used during the aggregation"""

    prepare_first: typing.Tuple[str, ...]
    reduce: typing.Tuple[str, ...]

    def __init__(self, *expressions, initial=_none, default=_none, where=None):
        super().__init__()
        self.where = None if where is None else self.ensure_conversion(where)
        self.expressions = tuple(
            self.ensure_conversion(expr)
            for expr in (
                self.expressions
                if hasattr(self, "expressions") and not expressions
                else expressions
            )
        )
        if default is not _none:
            self.default = default
        if hasattr(self, "default"):
            self.default = (
                self.ensure_conversion(self.default).call()
                if callable(self.default)
                else self.ensure_conversion(self.default)
            )

        if initial is not _none:
            self.initial = initial
        if hasattr(self, "initial"):
            self.initial = (
                self.ensure_conversion(self.initial).call()
                if callable(self.initial)
                else self.ensure_conversion(self.initial)
            )

        if not hasattr(self, "initial") and not hasattr(self, "default"):
            raise ValueError("both initial and default are none")
        if not hasattr(self, "initial") and not hasattr(self, "prepare_first"):
            raise ValueError("both initial and prepare_first are none")

    def _format_statements(
        self,
        statements,
        args,
        var_row,
        ctx,
    ):
        code_args = tuple(
            arg.gen_code_and_update_ctx(var_row, ctx) for arg in args
        )
        return [statement.format(*code_args) for statement in statements]

    def gen_reduce_code_block(
        self,
        var_agg_data_value: str,
        var_row: str,
        checksum_flag: int,
        ctx: dict,
    ) -> RBT:
        if hasattr(self, "initial"):
            reduce_initial = self._format_statements(
                self.reduce,
                (self.initial,) + self.expressions,
                var_row,
                ctx,
            )
        elif hasattr(self, "prepare_first"):
            reduce_initial = self._format_statements(
                self.prepare_first,
                self.expressions,
                var_row,
                ctx,
            )
        else:
            raise AssertionError

        reduce_two = self._format_statements(
            self.reduce,
            (EscapedString(var_agg_data_value),) + self.expressions,
            var_row,
            ctx,
        )

        block_cls = (
            ReduceBlock if self.where is None else ReduceConditionalBlock
        )
        kwargs = {
            "reduce_initial": reduce_initial,
            "reduce_two": reduce_two,
            "var_row": var_row,
            "var_agg_data_value": var_agg_data_value,
            "checksum_flag": checksum_flag,
            "unconditional_init": self.unconditional_init,
        }
        if self.where is not None:
            kwargs["condition_code"] = self.where.gen_code_and_update_ctx(
                var_row, ctx
            )

        return block_cls(**kwargs)


class Reduce(BaseReducer):
    """Defines the reduce operation, which is based on a callable /  to be used
    during the aggregation"""

    def __init__(
        self,
        to_call_with_2_args: typing.Union[typing.Callable, InlineExpr],
        *expressions: typing.Tuple[typing.Any, ...],
        initial: typing.Union[_None, typing.Callable, typing.Any] = _none,
        prepare_first: typing.Union[
            _None, typing.Callable, InlineExpr
        ] = _none,
        default: typing.Union[_None, typing.Callable, typing.Any] = _none,
        unconditional_init: bool = False,
        where=None,
    ):
        """
        Args:
          to_call_with_2_args: defines the reduce function/expression
          expressions: args to be passed to `to_call_with_2_args` after the
            aggregation value
          initial: defined the very first item to be passed to
            `to_call_with_2_args` item. If callable, then the result of a call
            is used.
          prepare_first: defines the reduce function/expression to be called
            on the first item to prepare it for subsequent calls of
            `to_call_with_2_args`
          default: defines the value to be returned when there was nothing to
            reduce in a group (e.g. the current reduce operation has filtered
            out some rows, while an adjacent reduce operation has got
            something to reduce, forming a group). If callable, then the result
            of a call is used.
          unconditional_init: tells whether the first call initializes the
            aggregation value OR there is a condition for that
        """
        super().__init__()
        self.where = None if where is None else self.ensure_conversion(where)
        self.to_call_with_2_args = self.ensure_conversion(to_call_with_2_args)
        self.expressions = tuple(
            self.ensure_conversion(expr) for expr in expressions
        )
        if initial is not _none:
            self.initial = (
                self.ensure_conversion(initial).call()
                if callable(initial)
                or isinstance(initial, NaiveConversion)
                and callable(initial.value)
                else self.ensure_conversion(initial)
            )
        if default is not _none:
            self.default = (
                self.ensure_conversion(default).call()
                if callable(default)
                or isinstance(default, NaiveConversion)
                and callable(default.value)
                else self.ensure_conversion(default)
            )

        if prepare_first is not _none:
            self.prepare_first = self.ensure_conversion(prepare_first)

        if not hasattr(self, "initial") and not hasattr(self, "default"):
            raise ValueError("both initial and default are none")
        if not hasattr(self, "initial") and not hasattr(self, "prepare_first"):
            raise ValueError("both initial and prepare_first are none")
        self.unconditional_init = unconditional_init

    def gen_reduce_code_block(
        self,
        var_agg_data_value: str,
        var_row: str,
        checksum_flag: int,
        ctx: dict,
    ) -> RBT:
        if hasattr(self, "initial"):
            initial_conversion = self.to_call_with_2_args.call_like(
                self.initial, *self.expressions
            )
        elif hasattr(self, "prepare_first"):
            initial_conversion = self.prepare_first.call_like(
                *self.expressions
            )
        else:
            raise AssertionError

        reduce_initial = [
            "%(result)s = {}".format(
                initial_conversion.gen_code_and_update_ctx(var_row, ctx),
            )
        ]
        reduce_two = [
            "%(result)s = {}".format(
                self.to_call_with_2_args.call_like(
                    EscapedString(var_agg_data_value),
                    *self.expressions,
                ).gen_code_and_update_ctx(var_row, ctx),
            )
        ]

        block_cls = (
            ReduceBlock if self.where is None else ReduceConditionalBlock
        )
        kwargs = {
            "reduce_initial": reduce_initial,
            "reduce_two": reduce_two,
            "var_row": var_row,
            "var_agg_data_value": var_agg_data_value,
            "checksum_flag": checksum_flag,
            "unconditional_init": self.unconditional_init,
        }
        if self.where is not None:
            kwargs["condition_code"] = self.where.gen_code_and_update_ctx(
                var_row, ctx
            )

        return block_cls(**kwargs)


class GroupBy(BaseConversion, typing.Generic[RBT]):
    """Generates the function which aggregates the data, grouping by
    conversions, specified in `__init__` method and returns list of items in a
    format defined by the parameter passed to ``aggregate`` method.

    If no group keys are passed, then it returns just a single value, defined
    by the parameter passed to ``aggregate`` method.

    Current optimizations:
     * piping like ``c.group_by(...).aggregate().pipe(...)`` won't run
       the aggregation twice, this is handled as 2 statements
     * using the same reduce clause twice (e.g. one used as an argument
       for some function calls) won't result in calculating this reduce twice
    """

    self_content_type = BaseConversion.ContentTypes.AGGREGATION

    def __init__(self, *by: typing.Tuple[BaseConversion, ...]):
        """Takes any number of conversions to group by

        Args:
          by (tuple): each item is to be wrapped with
            :py:obj:`ensure_conversion`.  Each is to resolve to a hashable
            object to allow using such tuples as keys. If nothing is passed,
            aggregate the input into a single object.
        """
        super().__init__()
        self.by = [self.ensure_conversion(by_) for by_ in by]
        self.agg_result: typing.Optional[BaseConversion] = None
        self.aggregate_mode = len(self.by) == 0
        self.filter_conversion = None
        self.filter_cast = None

    def aggregate(
        self, reducer: typing.Union[dict, list, set, tuple, BaseConversion]
    ) -> "GroupBy":
        """Takes the conversion which defines the desired output of
        aggregation"""
        self_clone = self.clone()
        self_clone.agg_result = self_clone.ensure_conversion(reducer)
        self_clone.contents = self_clone.contents & ~self.ContentTypes.REDUCER
        if isinstance(self_clone.agg_result, NaiveConversion):
            raise AssertionError("unexpected reducer type", type(reducer))
        return self_clone

    def filter(
        self, condition_conv, cast=BaseConversion._none
    ) -> "BaseConversion":
        if cast is self._none:
            cast = list
        if self.aggregate_mode or self.filter_conversion is not None:
            return super().filter(condition_conv, cast=cast)
        self_clone = self.clone()
        self_clone.filter_conversion = self_clone.ensure_conversion(
            condition_conv
        )
        self_clone.filter_cast = cast
        return self_clone

    def _gen_agg_data_container(self, container_name, number_of_reducers, ctx):
        attrs = ["v%d" % i for i in range(number_of_reducers)]

        code = Code()
        code.add_line(f"class {container_name}:", 1)
        code.add_line(
            "__slots__ = [{}]".format(",".join(f"'{attr}'" for attr in attrs)),
            0,
        )
        code.add_line("def __init__(self, _none=__none__):", 1)
        if attrs:
            for attr in attrs:
                code.add_line(f"self.{attr} = _none", 0)
        else:
            code.add_line("pass", 0)
        return self._code_to_converter(container_name, code.to_string(0), ctx)

    def _gen_code_and_update_ctx(self, code_input, ctx) -> str:
        if self.agg_result is None:
            raise AssertionError("aggregate hasn't been called")
        aggregate_mode = len(self.by) == 0
        suffix = self.gen_name("_", ctx, self)
        var_row = f"row{suffix}"
        var_signature = f"signature{suffix}"
        var_signature_to_agg_data = f"signature_to_agg_data{suffix}"
        var_agg_data = f"agg_data{suffix}"
        var_agg_data_cls = f"AggData{suffix}"

        signature_code_items = [
            by_.gen_code_and_update_ctx(var_row, ctx) for by_ in self.by
        ]
        replacements = {}

        # remembering code replacements for group by keys
        if len(signature_code_items) == 1:
            code_signature = signature_code_items[0]
            replacements[code_signature] = var_signature
        else:
            code_signature = f"({','.join(signature_code_items)},)"
            replacements.update(
                {
                    code: f"{var_signature}[{index}]"
                    for index, (code, by_) in enumerate(
                        zip(signature_code_items, self.by)
                    )
                }
            )

        # collecting inputs of reducers and reducers themselves
        with CodeGenerationOptionsCtx() as options:
            options.reducers_run_stage = "collecting_reducer_inputs"
            key = "_reducer_inputs_info"
            if key not in ctx:
                ctx[key] = []
            ctx[key].append(defaultdict(list))
            self.agg_result.gen_code_and_update_ctx(var_row, ctx)
            reducer_inputs_info = ctx[key].pop()

        def gen_agg_data_value(value_index):
            if aggregate_mode:
                return EscapedString(
                    f"{var_agg_data}_v{value_index}"
                ).gen_code_and_update_ctx("", ctx)
            else:
                return (
                    EscapedString(var_agg_data)
                    .attr(f"v{value_index}")
                    .gen_code_and_update_ctx("", ctx)
                )

        expected_checksum = 0
        var_agg_data_values = []

        blocks: typing.List[RBT] = []
        overwritten_reducer_inputs = {}
        # reusing same reducers, remembering code replacements for reducers
        for reducer_code_input, reducer in (
            (reducer_code_input, reducer)
            for reducer_code_input, reducers in reducer_inputs_info.items()
            for reducer in reducers
        ):
            reduce_block_index = len(blocks)
            checksum_flag = (
                1 << reduce_block_index if self.aggregate_mode else 0
            )
            var_agg_data_value = gen_agg_data_value(reduce_block_index)
            reduce_block = reducer.gen_reduce_code_block(
                var_agg_data_value,
                reducer_code_input,
                checksum_flag,
                ctx,
            )

            such_block_exists = False
            for index, block in enumerate(blocks):
                if reduce_block.same_as(block):
                    reduce_block_index = index
                    such_block_exists = True
                    break

            if not such_block_exists:
                blocks.append(reduce_block)
                var_agg_data_values.append(var_agg_data_value)
                expected_checksum |= checksum_flag

            overwritten_reducer_inputs[
                (reducer_code_input, reducer)
            ] = gen_agg_data_value(reduce_block_index)

        reduce_blocks: ReduceBlocks = ReduceBlocks()
        for block in blocks:
            reduce_blocks.add_block(block)

        # populates reducers with their code inputs
        with CodeGenerationOptionsCtx() as options:
            options.reducers_run_stage = "rendering_reducer_results"
            key = "_overwritten_reducer_inputs"
            if key not in ctx:
                ctx[key] = []
            ctx[key].append(overwritten_reducer_inputs)
            code_agg_result = self.agg_result.gen_code_and_update_ctx(
                var_row, ctx
            )
            ctx[key].pop()

        ctx.update({"defaultdict": defaultdict})

        if aggregate_mode:
            code_init_agg_vars = "{} = _none".format(
                " = ".join(var_agg_data_values)
            )
        else:
            ctx[var_agg_data_cls] = self._gen_agg_data_container(
                var_agg_data_cls, reduce_blocks.number, ctx
            )

        for code_from, code_to in replacements.items():
            code_agg_result = self.replace_word(
                code_agg_result, code_from, code_to
            )

        if self.count_words(code_agg_result, var_row):
            raise ConversionException(
                "neither group by key nor a field used in a reducer",
                code_agg_result,
            )
        if self.aggregate_mode:
            code_agg_result = f"    return {code_agg_result}"
        else:
            if self.filter_conversion is None:
                code_agg_result = (
                    f"    return [{code_agg_result} "
                    f"for {var_signature}, {var_agg_data} "
                    f"in {var_signature_to_agg_data}.items()]"
                )
            else:
                code_filtered_result = FilterConversion(
                    self.filter_conversion,
                    self.filter_cast,
                ).gen_code_and_update_ctx("result_", ctx)
                code_agg_result = (
                    f"    result_ = ({code_agg_result} "
                    f"for {var_signature}, {var_agg_data} "
                    f"in {var_signature_to_agg_data}.items())\n"
                    f"    filtered_result_ = {code_filtered_result}\n"
                ) + (
                    "    yield from filtered_result_"
                    if self.filter_cast is None
                    else "    return filtered_result_"
                )

        agg_template_kwargs = dict(
            code_args=self.get_args_def_code(as_kwargs=False),
            code_result=code_agg_result,
            var_row=var_row,
            expected_checksum=expected_checksum,
        )

        if self.aggregate_mode:
            converter_name = f"aggregate{suffix}"
            grouper_code = AGGREGATE_TEMPLATE.format(
                converter_name=converter_name,
                code_init_agg_vars=code_init_agg_vars,
                var_expected_checksum=ReduceBlock.var_expected_checksum,
                val_expected_checksum=expected_checksum,
                var_checksum=ReduceBlock.var_checksum,
                code_aggregate=reduce_blocks.gen_aggregate_code(
                    var_row=var_row,
                    var_checksum=ReduceBlock.var_checksum,
                    var_expected_checksum=expected_checksum,
                ).to_string(
                    base_indent_level=1,
                ),
                **agg_template_kwargs,
            )
        else:
            converter_name = f"group_by{suffix}"
            grouper_code = GROUPER_TEMPLATE.format(
                converter_name=converter_name,
                var_signature_to_agg_data=var_signature_to_agg_data,
                var_agg_data_cls=var_agg_data_cls,
                var_agg_data=var_agg_data,
                code_signature=code_signature,
                code_group_by=reduce_blocks.gen_group_by_code(
                    var_row=var_row,
                    var_agg_data=var_agg_data,
                    var_signature_to_agg_data=var_signature_to_agg_data,
                    code_signature=code_signature,
                ).to_string(base_indent_level=1),
                **agg_template_kwargs,
            )

        group_data_func = self._code_to_converter(
            converter_name=converter_name,
            code=grouper_code,
            ctx=ctx,
        )
        return CallFunc(
            group_data_func, GetItem(), *self.get_args_as_func_args()
        ).gen_code_and_update_ctx(code_input, ctx)


def Aggregate(*args, **kwargs) -> BaseConversion:
    """Shortcut for ``GroupBy().aggregate(*args, **kwargs)``"""
    return GroupBy().aggregate(*args, **kwargs)


class SumReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = ({0} or 0)",)
    reduce = ("%(result)s = {0} + ({1} or 0)",)
    default = 0
    unconditional_init = True


class SumOrNoneReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = {0}",)
    reduce = (
        "if {1} is None:",
        "    %(result)s = None",
        "elif {0} is not None:",
        "    %(result)s = {0} + {1}",
    )
    default = None
    unconditional_init = True


class MaxReducer(MultiStatementReducer):
    prepare_first = (
        "if {0} is not None:",
        "    %(result)s = {0}",
    )
    reduce = (
        "if {1} is not None and {1} > {0}:",
        "    %(result)s = {1}",
    )
    default = None


class MinReducer(MultiStatementReducer):
    prepare_first = (
        "if {0} is not None:",
        "    %(result)s = {0}",
    )
    reduce = (
        "if {1} is not None and {1} < {0}:",
        "    %(result)s = {1}",
    )
    default = None


class CountReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = 1",)
    reduce = ("%(result)s = {0} + 1",)
    default = 0
    unconditional_init = True


class CountDistinctReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = {{ {0} }}",)
    reduce = ("%(result)s.add({1})",)
    default = 0
    unconditional_init = True
    post_conversion = InlineExpr("{set_} and len({set_}) or 0").pass_args(
        set_=GetItem()
    )


class FirstReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = {0}",)
    reduce = ()
    default = None
    unconditional_init = True


class LastReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = {0}",)
    reduce = ("%(result)s = {1}",)
    default = None
    unconditional_init = True


class MaxRowReducer(MultiStatementReducer):
    """Reducer which finds an item with max value of the expression and returns
    this item"""

    prepare_first = (
        "if {0} is not None:",
        "    %(result)s = ({0}, %(row)s)",
    )
    reduce = (
        "if {1} is not None and {0}[0] < {1}:",
        "    %(result)s = ({1}, %(row)s)",
    )
    default = None
    post_conversion = GetItem(1)


class MinRowReducer(MaxRowReducer):
    reduce = (
        "if {1} is not None and {0}[0] > {1}:",
        "    %(result)s = ({1}, %(row)s)",
    )


class ArrayReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = [{0}]",)
    reduce = ("%(result)s.append({1})",)
    default = None
    unconditional_init = True


class ArrayDistinctReducer(MultiStatementReducer):
    prepare_first = ("%(result)s = {{ {0}: None }}",)
    reduce = ("%(result)s[{1}] = None",)
    post_conversion = InlineExpr("list({0})").pass_args(GetItem())
    default = None
    unconditional_init = True


class BaseDictReducer(MultiStatementReducer):
    """This reducer accepts 2 expressions:

    - the first one is used to calculate keys of the resulting dict
    - the second one is used to calculate values to be reduced and put as the
      final value, under the certain key.

    Effectively dict reducers allow for double grouping: the first one on the
    top level, the second one on DictReducer level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for backward compatibility with versions <= 0.8
        if len(self.expressions) == 1:
            expr = self.expressions[0]
            if isinstance(expr, (Tuple, List)) and len(expr.items) == 2:
                self.expressions = tuple(expr.items)
        if len(self.expressions) != 2:
            raise ValueError("invalid dict reducer input: two values expected")


class DictReducer(BaseDictReducer):
    prepare_first = ("%(result)s = {{ {0}: {1} }}",)
    reduce = ("%(result)s[{1}] = {2}",)
    default = None
    unconditional_init = True


class DictArrayReducer(BaseDictReducer):
    prepare_first = (
        "%(result)s = _d = defaultdict(list)",
        "_d[{0}].append({1})",
    )
    reduce = ("%(result)s[{1}].append({2})",)
    post_conversion = InlineExpr("dict({})").pass_args(GetItem())
    default = None
    unconditional_init = True


class DictArrayDistinctReducer(BaseDictReducer):
    """Dict reducer where dict values are lists of distinct values"""

    prepare_first = (
        "%(result)s = _d = defaultdict(dict)",
        "_d[{0}][{1}] = None",
    )
    reduce = ("%(result)s[{1}][{2}] = None",)
    post_conversion = InlineExpr(
        "{{k_: list(v_) for k_, v_ in {}.items()}}"
    ).pass_args(GetItem())
    default = None
    unconditional_init = True


class DictSumReducer(BaseDictReducer):
    prepare_first = (
        "%(result)s = _d = defaultdict(int)",
        "_d[{0}] = {1} or 0",
    )
    reduce = ("%(result)s[{1}] = {0}[{1}] + ({2} or 0)",)
    post_conversion = InlineExpr("dict({})").pass_args(GetItem())
    default = None
    unconditional_init = True


class DictSumOrNoneReducer(BaseDictReducer):
    """Dict reducer where dict values are either numbers or None if there's
    been at least one None value within the group"""

    prepare_first = ("%(result)s = _d = defaultdict(int)", "_d[{0}] = {1}")
    reduce = (
        "if {2} is None:",
        "    %(result)s[{1}] = None",
        "elif {0}[{1}] is not None:",
        "    %(result)s[{1}] = {0}[{1}] + {2}",
    )
    post_conversion = InlineExpr("dict({})").pass_args(GetItem())
    default = None
    unconditional_init = True


class DictMaxReducer(BaseDictReducer):
    prepare_first = (
        "if {1} is not None:",
        "    %(result)s = {{ {0}: {1} }}",
    )
    reduce = (
        "if {2} is not None and ({1} not in {0} or {2} > {0}[{1}]):",
        "    %(result)s[{1}] = {2}",
    )
    default = None


class DictMinReducer(DictMaxReducer):
    reduce = (
        "if {2} is not None and ({1} not in {0} or {2} < {0}[{1}]):",
        "    %(result)s[{1}] = {2}",
    )


class DictCountReducer(BaseDictReducer):
    prepare_first = ("%(result)s = {{ {0}: 1 }}",)
    reduce = (
        "if {1} not in {0}:",
        "    %(result)s[{1}] = 1",
        "else:",
        "    %(result)s[{1}] = {0}[{1}] + 1",
    )
    default = None
    unconditional_init = True


class DictCountDistinctReducer(BaseDictReducer):
    """Dict reducer where dict values are numbers of distinct values per
    group"""

    prepare_first = ("%(result)s = {{ {0}: {{ {1} }} }}",)
    reduce = (
        "if {1} not in {0}:",
        "    %(result)s[{1}] = {{ {2} }}",
        "else:",
        "    %(result)s[{1}].add({2})",
    )
    post_conversion = InlineExpr(
        "{{ k_: len(v_) for k_, v_ in {}.items() }}"
    ).pass_args(GetItem())
    default = None
    unconditional_init = True


class DictFirstReducer(BaseDictReducer):
    prepare_first = ("%(result)s = {{ {0}: {1} }}",)
    reduce = ("if {1} not in {0}:", "    %(result)s[{1}] = {2}")
    default = None
    unconditional_init = True


class DictLastReducer(BaseDictReducer):
    prepare_first = ("%(result)s = {{ {0}: {1} }}",)
    reduce = ("%(result)s[{1}] = {2}",)
    default = None
    unconditional_init = True


class AverageReducer(MultiStatementReducer):
    """
    Calculates the arithmetic mean or weighted mean.
    """

    def __init__(self, value, weight=1, **kwargs):
        super().__init__(value, weight, **kwargs)

    prepare_first = (
        "if {0} is not None:",
        "    %(result)s = ({1}, {0} * {1})",
    )
    reduce = (
        "if {1} is not None:",
        "    %(result)s = ({0}[0] + {2}, {0}[1] + {1} * {2})",
    )
    default = None
    post_conversion = GetItem(1) / GetItem(0)


class TopReducer(DictCountReducer):
    """
    Returns a list of the most frequent values.
    The resulting list is sorted in descending order of values frequency.
    """

    def __init__(self, k: int, key_conv, *args, **kwargs):
        super().__init__(key_conv, 1, *args, **kwargs)
        if not isinstance(k, int):
            raise TypeError("K must be an integer.")

        if k < 1:
            raise ValueError("K must be a positive integer greater than 0.")

        self.k = k

    @property
    def post_conversion(self):
        return InlineExpr(
            "[k for k,v in sorted((v,k) for k,v in {data}.items())[:-{k}:-1]]"
        ).pass_args(data=GetItem(), k=self.k + 1)


class ModeReducer(DictCountReducer):
    def __init__(self, conv, *args, **kwargs):
        super().__init__(conv, conv, *args, **kwargs)

    post_conversion = InlineExpr(
        "sorted(((v,k) for k,v in {data}.items()))[-1][1]"
    ).pass_args(data=GetItem())


class MedianReducer(ArrayReducer):
    post_conversion = CallFunc(statistics.median, GetItem())


class ReduceFuncs:
    """Exposes the list of reduce functions"""

    # pylint: disable=invalid-name

    #: Calculates the sum, skips false values
    Sum = SumReducer
    #: Calculates the sum, any ``None`` makes the total sum ``None``
    SumOrNone = SumOrNoneReducer

    #: Finds max value, skips ``None``
    Max = MaxReducer
    #: Finds a row with max value, skips ``None``
    MaxRow = MaxRowReducer

    #: Finds min value, skips ``None``
    Min = MinReducer
    #: Finds a row with min value, skips ``None``
    MinRow = MinRowReducer

    #: Counts objects
    Count = CountReducer
    #: Counts distinct values
    CountDistinct = CountDistinctReducer

    #: Stores the first value per group
    First = FirstReducer
    #: Stores the last value per group
    Last = LastReducer

    #: Calculates the arithmetic mean or weighted mean.
    Average = AverageReducer
    #: Calculates the median value.
    Median = MedianReducer
    #: Calculates the most common value.
    #: In case of multiple values, returns the last of them.
    Mode = ModeReducer
    #: Returns a list of the most frequent values.
    #: The resulting list is sorted in descending order of values frequency.
    TopK = TopReducer

    #: Aggregates values into array
    Array = ArrayReducer
    #: Aggregates distinct values into array, preserves order
    ArrayDistinct = ArrayDistinctReducer

    #: Aggregates values into dict; dict values are last values per group
    Dict = DictReducer
    #: Aggregates values into dict; dict values are lists of group values
    DictArray = DictArrayReducer
    #: Aggregates values into dict; dict values are lists of unique group
    #: values preserves order
    DictArrayDistinct = DictArrayDistinctReducer
    #: Aggregates values into dict; dict values are sums of group values,
    #: skipping ``None``
    DictSum = DictSumReducer
    #: Aggregates values into dict; dict values are sums of group values,
    #: any ``None`` makes the total sum ``None``
    DictSumOrNone = DictSumOrNoneReducer
    #: Aggregates values into dict; dict values are max group values
    DictMax = DictMaxReducer
    #: Aggregates values into dict; dict values are min group values
    DictMin = DictMinReducer
    #: Aggregates values into dict; dict values are numbers of values in groups
    DictCount = DictCountReducer
    #: Aggregates values into dict; dict values are numbers of unique values
    #: in groups
    DictCountDistinct = DictCountDistinctReducer
    #: Aggregates values into dict; dict values are first values per group
    DictFirst = DictFirstReducer
    #: Aggregates values into dict; dict values are last values per group
    DictLast = DictLastReducer
