"""Small helper so training fails fast and clearly when the shared MLflow server is unreachable.

Without this, an unreachable http(s) tracking URI makes MLflow retry silently for minutes and
then raise a long, confusing error half way through a training run.
"""
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def is_remote(uri):
    return bool(uri) and uri.lower().startswith(("http://", "https://"))


def ensure_server_reachable(uri, timeout=10):
    """Raise RuntimeError with a clear message if an http(s) MLflow server does not answer /health.

    Local URIs (sqlite:, file paths) are not checked. Returns True when the server answered.
    """
    if not is_remote(uri):
        return True
    url = uri.rstrip("/") + "/health"
    try:
        with urlopen(url, timeout=timeout) as response:
            if response.status == 200:
                return True
            reason = f"HTTP {response.status}"
    except HTTPError as error:
        reason = f"HTTP {error.code} {error.reason}"
    except (URLError, OSError) as error:
        reason = str(getattr(error, "reason", error))
    raise RuntimeError(
        f"MLflow server not reachable at {uri} ({reason}). Check that MLFLOW_TRACKING_URI is the "
        "current Elastic Beanstalk URL, that it starts with http://, and that the environment is Green."
    )
