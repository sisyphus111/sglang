import socket
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints.http_server import (
    _setup_and_run_http_server,
    launch_server,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPreboundHttpSocket(unittest.TestCase):
    def _server_args(self):
        return SimpleNamespace(
            enable_metrics=False,
            tokenizer_worker_num=1,
            api_key=None,
            admin_api_key=None,
            enable_http2=False,
            enable_ssl_refresh=False,
            ssl_certfile=None,
            ssl_keyfile=None,
            ssl_ca_certs=None,
            ssl_keyfile_password=None,
            host="127.0.0.1",
            port=30000,
            fastapi_root_path="",
            log_level_http=None,
            log_level="info",
        )

    def test_uvicorn_receives_the_exact_prebound_socket(self):
        http_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server = MagicMock(started=True)
        with patch("sglang.srt.entrypoints.http_server.set_global_state"), patch(
            "sglang.srt.entrypoints.http_server.app_has_admin_force_endpoints",
            return_value=False,
        ), patch(
            "sglang.srt.entrypoints.http_server.set_uvicorn_logging_configs"
        ), patch(
            "sglang.srt.entrypoints.http_server.uvicorn.Config"
        ) as config, patch(
            "sglang.srt.entrypoints.http_server.uvicorn.Server",
            return_value=server,
        ):
            _setup_and_run_http_server(
                self._server_args(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                [{}],
                None,
                http_socket=http_socket,
            )

        config.assert_called_once()
        server.run.assert_called_once_with(sockets=[http_socket])
        http_socket.close()

    def test_uvicorn_startup_failure_remains_visible(self):
        http_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server = MagicMock(started=False)
        with patch("sglang.srt.entrypoints.http_server.set_global_state"), patch(
            "sglang.srt.entrypoints.http_server.app_has_admin_force_endpoints",
            return_value=False,
        ), patch(
            "sglang.srt.entrypoints.http_server.set_uvicorn_logging_configs"
        ), patch(
            "sglang.srt.entrypoints.http_server.uvicorn.Config"
        ), patch(
            "sglang.srt.entrypoints.http_server.uvicorn.Server",
            return_value=server,
        ):
            with self.assertRaisesRegex(RuntimeError, "Uvicorn failed to start"):
                _setup_and_run_http_server(
                    self._server_args(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    [{}],
                    None,
                    http_socket=http_socket,
                )
        http_socket.close()

    def test_unsupported_socket_mode_fails_before_engine_launch(self):
        http_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_args = ServerArgs(model_path="dummy", tokenizer_worker_num=2)
        with patch(
            "sglang.srt.entrypoints.http_server.Engine._launch_subprocesses"
        ) as launch:
            with self.assertRaisesRegex(ValueError, "one tokenizer worker"):
                launch_server(server_args, http_socket=http_socket)
        launch.assert_not_called()
        http_socket.close()


if __name__ == "__main__":
    unittest.main()
