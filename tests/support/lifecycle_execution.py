"""Complete lifecycle execution port for deterministic test gateways."""

from __future__ import annotations

from typing import Any, Sequence

from nautical_core.integration_models import (
    ChildImportPayload,
    MutationOperation,
    MutationOutcome,
    MutationOutcomeKind,
    MutationPostcondition,
    MutationRequest,
)


class LifecycleExecutionFixture:
    """Adapt single-mutation test doubles to the explicit lifecycle port.

    Batch verification is intentionally synthetic; production behavior is
    covered by lifecycle tests that use TaskwarriorMutationService.
    """

    def __init__(self, mutation_gateway: Any) -> None:
        self._mutation_gateway = mutation_gateway

    def apply(self, request: MutationRequest) -> MutationOutcome:
        return self._mutation_gateway.apply(request)

    def compensate_imported_child(self, request: MutationRequest) -> MutationOutcome:
        return MutationOutcome(
            MutationOperation.CHILD_COMPENSATION,
            MutationOutcomeKind.MANUAL_REVIEW,
            request.guard,
            reason="test fixture does not model child compensation",
        )

    def apply_lifecycle_unverified(self, request: MutationRequest) -> MutationOutcome:
        return self.apply(request)

    def apply_lifecycle_children_unverified(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        return {
            str(request.payload.child_uuid).lower(): self.apply(request)
            for request in requests
        }

    def verify_lifecycle_children(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        return {
            str(request.payload.child_uuid).lower(): self._verified(request, MutationPostcondition.CHILD_IMPORTED)
            for request in requests
        }

    def verify_lifecycle_parents(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        return {
            request.guard.task_uuid.lower(): self._verified(request, MutationPostcondition.PARENT_LINKED)
            for request in requests
        }

    def preflight_lifecycle_batch(
        self,
        _payloads: Sequence[ChildImportPayload],
        *,
        parent_expectations: Sequence[tuple[str, str]] = (),
    ) -> None:
        del parent_expectations
        return None

    @staticmethod
    def _verified(request: MutationRequest, postcondition: MutationPostcondition) -> MutationOutcome:
        return MutationOutcome(
            request.operation,
            MutationOutcomeKind.APPLIED,
            request.guard,
            (postcondition,),
        )
