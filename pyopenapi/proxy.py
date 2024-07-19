import json
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar

import aiohttp
from strong_typing.serialization import json_to_object, object_to_json

from .operations import (
    EndpointOperation,
    HTTPMethod,
    get_endpoint_operations,
    get_signature,
)


async def make_request(
    http_method: HTTPMethod,
    server: str,
    path: str,
    query: Dict[str, str],
    data: Optional[str],
) -> Tuple[int, str]:
    "Makes an asynchronous HTTP request and returns the response."

    headers = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"

    async with aiohttp.ClientSession(server) as session:
        if http_method is HTTPMethod.GET:
            fn = session.get
        elif http_method is HTTPMethod.POST:
            fn = session.post
        elif http_method is HTTPMethod.PUT:
            fn = session.put
        elif http_method is HTTPMethod.DELETE:
            fn = session.delete
        elif http_method is HTTPMethod.PATCH:
            fn = session.patch
        else:
            raise NotImplementedError(f"unknown HTTP method: {http_method}")

        async with fn(path, headers=headers, params=query, data=data) as resp:
            body = await resp.text()
            return resp.status, body


class ProxyInvokeError(RuntimeError):
    pass


class EndpointProxy:
    "The HTTP REST proxy class for an endpoint."

    url: str

    def __init__(self, base_url: str):
        self.base_url = base_url


class OperationProxy:
    """
    The HTTP REST proxy class for an endpoint operation.

    Extracts operation parameters from the Python API signature such as route, path and query parameters and request
    payload, builds an HTTP request, and processes the HTTP response.
    """

    def __init__(self, op: EndpointOperation):
        self.op = op
        self.sig = get_signature(op.func_ref)

    async def __call__(
        self, endpoint_proxy: EndpointProxy, *args: Any, **kwargs: Any
    ) -> Any:
        "Invokes an API operation via HTTP REST."

        ba = self.sig.bind(self, *args, **kwargs)

        # substitute parameters in URL path
        route = self.op.get_route()
        path = route.format_map(
            {name: ba.arguments[name] for name, _type in self.op.path_params}
        )

        # gather URL query parameters
        query = {name: str(ba.arguments[name]) for name, _type in self.op.query_params}

        # assemble request body
        if len(self.op.request_params) > 0:
            value = {name: ba.arguments[name] for name, _type in self.request_params}
            data = json.dumps(
                object_to_json(value),
                check_circular=False,
                indent=None,
                separators=(",", ":"),
            )
        else:
            data = None

        # make HTTP request
        status, response = await make_request(
            self.op.http_method, endpoint_proxy.base_url, path, query, data
        )

        # process HTTP response
        if response:
            try:
                s = json.loads(response)
            except json.JSONDecodeError:
                raise ProxyInvokeError(
                    f"response body is not well-formed JSON:\n{response}"
                )

            return json_to_object(self.op.response_type, s)
        else:
            return None


def _get_operation_proxy(op: EndpointOperation) -> Callable[..., Any]:
    "Wraps an operation into a function that calls the corresponding HTTP REST API operation."

    operation_proxy = OperationProxy(op)

    async def _operation_proxy_fn(
        self: EndpointProxy, *args: Any, **kwargs: Any
    ) -> Any:
        return await operation_proxy(self, *args, **kwargs)

    return _operation_proxy_fn


T = TypeVar("T")


def make_proxy_class(api: Type[T]) -> Type[T]:
    """
    Creates a proxy class for calling an HTTP REST API.

    :param api: The endpoint (as a Python class) that defines operations.
    """

    ops = get_endpoint_operations(api)
    properties = {op.func_name: _get_operation_proxy(op) for op in ops}
    proxy = type(f"{api.__name__}Proxy", (api, EndpointProxy), properties)
    return proxy
