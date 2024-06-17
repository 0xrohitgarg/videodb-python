import requests
import os

from typing import Optional
from requests import HTTPError


from videodb._constants import (
    ApiPath,
)

from videodb.exceptions import (
    VideodbError,
)


STREAMING_API = os.getenv(
    "STREAMING_API", "https://vcmgicsv1d.execute-api.us-east-1.amazonaws.com"
)
SEGMENT_DURATION = os.getenv("SEGMENT_DURATION", 1)
SEGMENT_TYPE = os.getenv("SEGMENT_TYPE", "fmp4")


def upload(
    _connection,
    file_path: str = None,
    url: str = None,
    media_type: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    callback_url: Optional[str] = None,
) -> dict:
    if not file_path and not url:
        raise VideodbError("Either file_path or url is required")
    if file_path and url:
        raise VideodbError("Only one of file_path or url is allowed")

    if file_path:
        try:
            name = file_path.split("/")[-1].split(".")[0] if not name else name
            upload_url_data = _connection.get(
                path=f"{ApiPath.collection}/{_connection.collection_id}/{ApiPath.upload_url}",
                params={"name": name},
            )
            upload_url = upload_url_data.get("upload_url")
            with open(file_path, "rb") as file:
                files = {"file": (name, file)}
                response = requests.post(upload_url, files=files)
                response.raise_for_status()
                url = upload_url

        except FileNotFoundError as e:
            raise VideodbError("File not found", cause=e)

        except HTTPError as e:
            raise VideodbError("Error while uploading file", cause=e)

    upload_data = _connection.post(
        path=f"{ApiPath.collection}/{_connection.collection_id}/{ApiPath.upload}",
        data={
            "url": url,
            "name": name,
            "description": description,
            "callback_url": callback_url,
            "media_type": media_type,
        },
    )

    # Temporary test code for streaming
    try:
        media_id = upload_data.get("id", "")
        if media_id.startswith("m-"):
            streaming_data = requests.post(
                f"{STREAMING_API}/upload",
                json={
                    "url": url,
                    "media_id": media_id,
                    "user_id": _connection.user_id,
                    "segment_duration": SEGMENT_DURATION,
                    "segment_type": SEGMENT_TYPE,
                },
            )
            streaming_data.raise_for_status()
            streaming_data = streaming_data.json()
            upload_data["stream_url"] = streaming_data.get("stream_url")
    except HTTPError as e:
        raise VideodbError("Error while uploading file", cause=e)
    return upload_data
