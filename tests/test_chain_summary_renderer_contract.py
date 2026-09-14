from __future__ import annotations

import unittest
from unittest.mock import patch

from nautical_core.modify_chain_summary import ChainSummaryRenderServices, render_chain_summary_with_services


class ChainSummaryRendererContractTests(unittest.TestCase):
    def test_service_bundle_delegates_to_renderer(self) -> None:
        calls = []
        services = ChainSummaryRenderServices(
            export_sorted_chain=lambda *_: [], root_uuid_from=lambda value: value.get("uuid"),
            short_uuid=lambda value: str(value or "")[:4], format_root_and_age=lambda *_: "root",
            kind_rows=lambda *_: None, span_fields=lambda *_args, **_kwargs: (None, None, "–"),
            stats_rows=lambda *_: None, limits_row=lambda *_: None,
            last_n_timeline_rows=lambda *_: [], format_rows=lambda rows: rows,
            coerce_int=lambda value, default: int(value or default), format_local=lambda value: str(value),
            max_chain_walk=10, panel=lambda *args, **kwargs: calls.append((args, kwargs)), diagnostic=lambda _: None,
        )
        with patch("nautical_core.modify_chain_summary.render_chain_summary") as renderer:
            render_chain_summary_with_services({"uuid": "u", "chainID": "c"}, "done", None, None, services=services)
        renderer.assert_called_once()
        self.assertIs(renderer.call_args.kwargs["services"], services)


if __name__ == "__main__":
    unittest.main()
