from dataclasses import dataclass
from sglang.srt.environ import envs
from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDraftProxyThread,
    CppDraftTailBuffer,
    CppTokenSyncThread,
)
from sglang.srt.speculative.draft_proxy import DraftProxyThread
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.srt.speculative.token_sync_thread import TokenSyncThread


@dataclass(frozen=True)
class DecoupledSpecTransport:
    """Selected transport implementation and its data-plane capability."""

    draft_tail_buffer_cls: type
    draft_proxy_thread_cls: type
    token_sync_thread_cls: type
    supports_native_rows: bool


_CPP_TRANSPORT = DecoupledSpecTransport(
    draft_tail_buffer_cls=CppDraftTailBuffer,
    draft_proxy_thread_cls=CppDraftProxyThread,
    token_sync_thread_cls=CppTokenSyncThread,
    supports_native_rows=True,
)
_PYTHON_TRANSPORT = DecoupledSpecTransport(
    draft_tail_buffer_cls=DraftTailBuffer,
    draft_proxy_thread_cls=DraftProxyThread,
    token_sync_thread_cls=TokenSyncThread,
    supports_native_rows=False,
)


def get_decoupled_spec_transport() -> DecoupledSpecTransport:
    if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get():
        return _CPP_TRANSPORT
    return _PYTHON_TRANSPORT
