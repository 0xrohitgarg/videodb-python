"""Http client Module."""

import logging
import requests
import backoff

from typing import (
    Callable,
    Optional,
)
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from videodb._constants import (
    HttpClientDefaultValues,
    Status,
)
from videodb.exceptions import (
    AuthenticationError,
    InvalidRequestError,
)

logger = logging.getLogger(__name__)


class HttpClient:
    """Http client for making requests"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_retries: Optional[int] = HttpClientDefaultValues.max_retries,
    ) -> None:
        """Create a new http client instance

        :param str api_key: The api key to use for authentication
        :param str base_url: The base url to use for the api
        :param int max_retries: (optional) The maximum number of retries to make for a request
        """
        self.session = requests.Session()

        retries = Retry(
            total=max_retries,
            backoff_factor=HttpClientDefaultValues.backoff_factor,
            status_forcelist=HttpClientDefaultValues.status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {"x-access-token": api_key, "Content-Type": "application/json"}
        )
        self.base_url = base_url
        logger.debug(f"Initialized http client with base url: {self.base_url}")

    def _make_request(
        self,
        method: Callable[..., requests.Response],
        path: str,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ):
        """Make a request to the api

        :param Callable method: The method to use for the request
        :param str path: The path to make the request to
        :param str base_url: (optional) The base url to use for the request
        :param dict headers: (optional) The headers to use for the request
        :param kwargs: The keyword arguments to pass to the request method
        :return: json response from the request
        """
        try:
            url = f"{base_url or self.base_url}/{path}"
            timeout = kwargs.pop("timeout", HttpClientDefaultValues.timeout)
            request_headers = {**self.session.headers, **(headers or {})}
            response = method(url, headers=request_headers, timeout=timeout, **kwargs)
            response.raise_for_status()
            return self._parse_response(response)

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)

    def _handle_request_error(self, e: requests.exceptions.RequestException) -> None:
        """Handle request errors"""

        if isinstance(e, requests.exceptions.HTTPError):
            try:
                error_message = e.response.json().get("message", "Unknown error")
            except ValueError:
                error_message = e.response.text

            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Error: {error_message}", e.response
                ) from None
            else:
                raise InvalidRequestError(
                    f"Invalid request: {error_message}", e.response
                ) from None

        elif isinstance(e, requests.exceptions.RetryError):
            raise InvalidRequestError(
                "Invalid request: Max retries exceeded", e.response
            ) from None

        elif isinstance(e, requests.exceptions.Timeout):
            raise InvalidRequestError(
                "Invalid request: Request timed out", e.response
            ) from None

        elif isinstance(e, requests.exceptions.ConnectionError):
            raise InvalidRequestError(
                "Invalid request: Connection error", e.response
            ) from None

        else:
            raise InvalidRequestError(
                f"Invalid request: {str(e)}", e.response
            ) from None

    @backoff.on_exception(backoff.expo, Exception, max_time=500)
    def _get_output(self, url: str):
        """Get the output from an async request"""
        response_json = self.session.get(url).json()
        if (
            response_json.get("status") == Status.in_progress
            or response_json.get("status") == Status.processing
        ):
            logger.debug("Waiting for processing to complete")
            raise Exception("Stuck on processing status") from None
        return response_json.get("response") or response_json

    def _parse_response(self, response: requests.Response):
        """Parse the response from the api"""
        try:
            response_json = response.json()
            if (
                response_json.get("status") == Status.processing
                and response_json.get("request_type", "sync") == "async"
            ):
                return None
            elif (
                response_json.get("status") == Status.processing
                and response_json.get("request_type", "sync") == "sync"
            ):
                response_json = self._get_output(
                    response_json.get("data").get("output_url")
                )
                if response_json.get("success"):
                    return response_json.get("data")
                else:
                    raise InvalidRequestError(
                        f"Invalid request: {response_json.get('message')}", response
                    ) from None

            elif response_json.get("success"):
                return response_json.get("data")

            else:
                raise InvalidRequestError(
                    f"Invalid request: {response_json.get('message')}", response
                ) from None

        except ValueError:
            raise InvalidRequestError(
                f"Invalid request: {response.text}", response
            ) from None

    def get(self, path: str, **kwargs) -> requests.Response:
        """Make a get request"""
        return self._make_request(self.session.get, path, **kwargs)

    def post(self, path: str, data=None, **kwargs) -> requests.Response:
        """Make a post request"""
        return self._make_request(self.session.post, path, json=data, **kwargs)

    def put(self, path: str, data=None, **kwargs) -> requests.Response:
        """Make a put request"""
        return self._make_request(self.session.put, path, json=data, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        """Make a delete request"""
        return self._make_request(self.session.delete, path, **kwargs)

    def patch(self, path: str, data=None, **kwargs) -> requests.Response:
        """Make a patch request"""
        return self._make_request(self.session.patch, path, json=data, **kwargs)
