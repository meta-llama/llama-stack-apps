from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence, Set
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import pandas as pd
from gradio_client.documentation import document

from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Events

if TYPE_CHECKING:
    from gradio.components import Timer


class PlotData(GradioModel):
    columns: list[str]
    data: list[list[Any]]
    datatypes: dict[str, Literal["quantitative", "nominal", "temporal"]]
    mark: str

from gradio.events import Dependency

class NativePlot(Component):
    """
    Creates a native Gradio plot component to display data from a pandas DataFrame. Supports interactivity and updates.

    Demos: native_plots
    """

    EVENTS = [Events.select, Events.double_click]

    def __init__(
        self,
        value: pd.DataFrame | Callable | None = None,
        x: str | None = None,
        y: str | None = None,
        *,
        color: str | None = None,
        title: str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_title: str | None = None,
        x_bin: str | float | None = None,
        y_aggregate: Literal["sum", "mean", "median", "min", "max", "count"]
        | None = None,
        color_map: dict[str, str] | None = None,
        x_lim: list[float] | None = None,
        y_lim: list[float] | None = None,
        x_label_angle: float = 0,
        y_label_angle: float = 0,
        x_axis_labels_visible: bool = True,
        caption: str | None = None,
        sort: Literal["x", "y", "-x", "-y"] | list[str] | None = None,
        tooltip: Literal["axis", "none", "all"] | list[str] = "axis",
        height: int | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        every: Timer | float | None = None,
        inputs: Component | Sequence[Component] | Set[Component] | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: The pandas dataframe containing the data to display in the plot.
            x: Column corresponding to the x axis. Column can be numeric, datetime, or string/category.
            y: Column corresponding to the y axis. Column must be numeric.
            color: Column corresponding to series, visualized by color. Column must be string/category.
            title: The title to display on top of the chart.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_title: The title given to the color legend. By default, uses the value of color parameter.
            x_bin: Grouping used to cluster x values. If x column is numeric, should be number to bin the x values. If x column is datetime, should be string such as "1h", "15m", "10s", using "s", "m", "h", "d" suffixes.
            y_aggregate: Aggregation function used to aggregate y values, used if x_bin is provided or x is a string/category. Must be one of "sum", "mean", "median", "min", "max".
            color_map: Mapping of series to color names or codes. For example, {"success": "green", "fail": "#FF8888"}.
            height: The height of the plot in pixels.
            x_lim: A tuple or list containing the limits for the x-axis, specified as [x_min, x_max]. If x column is datetime type, x_lim should be timestamps.
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            x_label_angle: The angle of the x-axis labels in degrees offset clockwise.
            y_label_angle: The angle of the y-axis labels in degrees offset clockwise.
            x_axis_labels_visible: Whether the x-axis labels should be visible. Can be hidden when many x-axis labels are present.
            caption: The (optional) caption to display below the plot.
            sort: The sorting order of the x values, if x column is type string/category. Can be "x", "y", "-x", "-y", or list of strings that represent the order of the categories.
            tooltip: The tooltip to display when hovering on a point. "axis" shows the values for the axis columns, "all" shows all column values, and "none" shows no tooltips. Can also provide a list of strings representing columns to show in the tooltip, which will be displayed along with axis values.
            height: The height of the plot in pixels.
            label: The (optional) label to display on the top left corner of the plot.
            show_label: Whether the label should be displayed.
            container: If True, will place the component in a container - providing some extra padding around the border.
            scale: relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.
            every: Continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            inputs: Components that are used as inputs to calculate `value` if `value` is a function (has no effect otherwise). `value` is recalculated any time the inputs change.
            visible: Whether the plot should be visible.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
            key: if assigned, will be used to assume identity across a re-render. Components that have the same key across a re-render will have their value preserved.
        """
        self.x = x
        self.y = y
        self.color = color
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.color_title = color_title
        self.x_bin = x_bin
        self.y_aggregate = y_aggregate
        self.color_map = color_map
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_label_angle = x_label_angle
        self.y_label_angle = y_label_angle
        self.x_axis_labels_visible = x_axis_labels_visible
        self.caption = caption
        self.sort = sort
        self.tooltip = tooltip
        self.height = height

        if label is None and show_label is None:
            show_label = False
        super().__init__(
            value=value,
            label=label,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            every=every,
            inputs=inputs,
        )
        for key_, val in kwargs.items():
            if key_ == "color_legend_title":
                self.color_title = val
            if key_ in [
                "stroke_dash",
                "overlay_point",
                "x_label_angle",
                "y_label_angle",
                "interactive",
                "show_actions_button",
                "color_legend_title",
                "width",
            ]:
                warnings.warn(
                    f"Argument '{key_}' has been deprecated.", DeprecationWarning
                )

    def get_block_name(self) -> str:
        return "nativeplot"

    def get_mark(self) -> str:
        return "native"

    def preprocess(self, payload: PlotData | None) -> PlotData | None:
        """
        Parameters:
            payload: The data to display in a line plot.
        Returns:
            The data to display in a line plot.
        """
        return payload

    def postprocess(self, value: pd.DataFrame | dict | None) -> PlotData | None:
        """
        Parameters:
            value: Expects a pandas DataFrame containing the data to display in the line plot. The DataFrame should contain at least two columns, one for the x-axis (corresponding to this component's `x` argument) and one for the y-axis (corresponding to `y`).
        Returns:
            The data to display in a line plot, in the form of an AltairPlotData dataclass, which includes the plot information as a JSON string, as well as the type of plot (in this case, "line").
        """
        # if None or update
        if value is None or isinstance(value, dict):
            return value

        def get_simplified_type(dtype):
            if pd.api.types.is_numeric_dtype(dtype):
                return "quantitative"
            elif pd.api.types.is_string_dtype(
                dtype
            ) or pd.api.types.is_categorical_dtype(dtype):
                return "nominal"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                return "temporal"
            else:
                raise ValueError(f"Unsupported data type: {dtype}")

        split_json = json.loads(value.to_json(orient="split", date_unit="ms"))
        datatypes = {
            col: get_simplified_type(value[col].dtype) for col in value.columns
        }
        return PlotData(
            columns=split_json["columns"],
            data=split_json["data"],
            datatypes=datatypes,
            mark=self.get_mark(),
        )

    def example_payload(self) -> Any:
        return None

    def example_value(self) -> Any:
        import pandas as pd

        return pd.DataFrame({self.x: [1, 2, 3], self.y: [4, 5, 6]})

    def api_info(self) -> dict[str, Any]:
        return {"type": {}, "description": "any valid json"}
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer

    
    def select(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, the endpoint will be exposed in the api docs as an unnamed endpoint, although this behavior will be changed in Gradio 4.0. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
        
        """
        ...
    
    def double_click(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, the endpoint will be exposed in the api docs as an unnamed endpoint, although this behavior will be changed in Gradio 4.0. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
        
        """
        ...


@document()
class BarPlot(NativePlot):
    """
    Creates a bar plot component to display data from a pandas DataFrame.

    Demos: bar_plot_demo
    """

    def get_block_name(self) -> str:
        return "nativeplot"

    def get_mark(self) -> str:
        return "bar"
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer


@document()
class LinePlot(NativePlot):
    """
    Creates a line plot component to display data from a pandas DataFrame.

    Demos: line_plot_demo
    """

    def get_block_name(self) -> str:
        return "nativeplot"

    def get_mark(self) -> str:
        return "line"
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer


@document()
class ScatterPlot(NativePlot):
    """
    Creates a scatter plot component to display data from a pandas DataFrame.

    Demos: scatter_plot_demo
    """

    def get_block_name(self) -> str:
        return "nativeplot"

    def get_mark(self) -> str:
        return "point"
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer