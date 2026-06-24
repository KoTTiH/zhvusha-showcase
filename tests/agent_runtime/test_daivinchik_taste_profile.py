"""Daivinchik taste-profile Agent Runtime worker contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextPack
from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError, ToolGateway
from src.llm.protocols import LLMResponse, LLMUsage, LLMVisionRequest


def _job(context_pack: ContextPack, *, profile: Any | None = None) -> AgentJob:
    from src.agent_runtime.profiles import DAIVINCHIK_TASTE_PROFILE_READONLY

    return AgentJob.new(
        owner_user_id=12345,
        chat_id=12345,
        source_message_id="msg-daivinchik",
        fingerprint="daivinchik-profile-test",
        kind="daivinchik_taste_profile",
        profile=profile or DAIVINCHIK_TASTE_PROFILE_READONLY,
        context_pack=context_pack,
        status=AgentJobStatus.RUNNING,
    )


def _approved_pack(payload: dict[str, Any]) -> ContextPack:
    return ContextPack(
        user_request=json.dumps(payload),
        metadata={
            "agent_tool_approval_id": "daivinchik-live-test",
            "agent_tool_approval_capabilities": (
                "telegram_mcp_daivinchik_button,"
                "telegram_mcp_daivinchik_reply_button,"
                "telegram_mcp_daivinchik_notify,"
                "telegram_mcp_daivinchik_forward_liked_profile"
            ),
        },
    )


class FakeVision:
    def __init__(self) -> None:
        self.calls: list[LLMVisionRequest] = []

    async def describe_images(self, request: LLMVisionRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text="темные волосы, спокойный минималистичный стиль, фото в зеркале",
            model="fake-vision",
            usage=LLMUsage(),
        )


class FakeFrameExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    async def extract_first_frame(self, video_path: Path, frame_path: Path) -> None:
        self.calls.append((video_path, frame_path))
        frame_path.write_bytes(b"fake-frame")


class FakeTerminalVision:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    async def describe_image_file(
        self,
        image_path: Path,
        *,
        prompt: str,
        caller: str,
    ) -> str:
        del prompt, caller
        self.calls.append(image_path)
        return (
            "лицо видно, face_match:strong, естественный cute вайб, очки, "
            "домашняя обстановка"
        )


class _FakeCodexVisionProcess:
    def __init__(self, *, returncode: int, output_path: Path, final: str) -> None:
        self.returncode = returncode
        self._output_path = output_path
        self._final = final
        self.stdin: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.stdin = input
        self._output_path.write_text(self._final, encoding="utf-8")
        return b"stdout", b""

    def kill(self) -> None:
        return None

    async def wait(self) -> int:
        return self.returncode


class _FakeProfileClassifierProcess:
    def __init__(self, *, returncode: int, output_path: Path, final: str) -> None:
        self.returncode = returncode
        self._output_path = output_path
        self._final = final
        self.stdin: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.stdin = input
        self._output_path.write_text(self._final, encoding="utf-8")
        return b"stdout", b""

    def kill(self) -> None:
        return None

    async def wait(self) -> int:
        return self.returncode


def test_card_grouping_deduplicates_profiles_and_keeps_action_signal() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        normalize_history_to_cards,
    )

    raw_history = {
        "messages": [
            {
                "id": 1,
                "text": "Аня, 22\nМосква\nсырой приватный текст анкеты",
                "media": [{"id": "photo-1", "type": "photo"}],
            },
            {"id": 2, "text": "❤️"},
            {
                "id": 3,
                "text": "Аня, 22\nМосква\nсырой приватный текст анкеты",
                "media": [{"id": "photo-1", "type": "photo"}],
            },
            {"id": 4, "text": "Катя, 27\nСПб\nспорт и вечеринки"},
            {"id": 5, "text": "👎"},
        ]
    }

    cards = normalize_history_to_cards(raw_history)

    assert len(cards) == 2
    assert cards[0].action == "positive"
    assert cards[0].age == 22
    assert cards[0].city == "Москва"
    assert cards[0].text_hash
    assert cards[1].action == "negative"
    assert cards[1].age == 27


def test_card_grouping_extracts_under18_age_for_single_age_gate() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        normalize_history_to_cards,
    )

    cards = normalize_history_to_cards(
        {
            "messages": [
                {
                    "id": 17,
                    "text": "Маша, 17, Москва\nкофе и прогулки",
                    "media": [{"id": "photo-17", "type": "photo"}],
                }
            ]
        }
    )

    assert cards[0].age == 17
    assert cards[0].city == "Москва"


def test_history_parser_handles_nested_mcp_result_and_line_format() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        count_history_messages,
        normalize_history_to_cards,
    )

    nested = {
        "result": json.dumps(
            {
                "results": [
                    {"id": 2118, "sender": "Жвуша", "text": "👎"},
                    {"id": 2117, "sender": "Дайвинчик", "text": "[empty]"},
                    {
                        "id": 2116,
                        "sender": "Дайвинчик",
                        "text": "Гоар, 18, Москва – 🇬🇪🇦🇲🐈⬛",
                    },
                ]
            },
            ensure_ascii=False,
        )
    }
    line_format = (
        "ID: 2118 | Жвуша | Date: 2026-05-15 | Message: 👎\n"
        "ID: 2117 | Дайвинчик | Date: 2026-05-15 | Message: [empty]\n"
        "ID: 2116 | Дайвинчик | Date: 2026-05-15 | "
        "Message: Гоар, 18, Москва – 🇬🇪🇦🇲🐈⬛"
    )

    nested_cards = normalize_history_to_cards(nested)
    line_cards = normalize_history_to_cards(line_format)

    assert count_history_messages(nested) == 3
    assert nested_cards[0].age == 18
    assert nested_cards[0].city.lower() == "москва"
    assert nested_cards[0].action == "negative"
    assert len(nested_cards[0].media_refs) == 1
    assert count_history_messages(line_format) == 3
    assert line_cards[0].age == 18
    assert line_cards[0].action == "negative"
    assert len(line_cards[0].media_refs) == 1


def test_downloaded_path_handles_upstream_json_result_phrase(tmp_path: Path) -> None:
    from src.agent_runtime.workers.daivinchik_profile import _downloaded_path

    media_path = tmp_path / "media.jpg"
    media_path.write_bytes(b"fake")
    raw_result = json.dumps({"result": f"Media downloaded to {media_path}."})

    assert _downloaded_path(raw_result, temp_dir=tmp_path) == media_path


def test_media_kind_inference_does_not_treat_photo_video_sizes_as_video() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _infer_media_kind

    photo_info = "MessageMediaPhoto(photo=Photo(...), video_sizes=[])"
    video_info = "MessageMediaDocument(document=Document(mime_type='video/mp4'))"

    assert _infer_media_kind("unknown", photo_info) == "photo"
    assert _infer_media_kind("unknown", video_info) == "video"


def test_autolike_decision_likes_current_card_from_existing_taste_rules() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("101",),
        text_hash="text",
        content_hash="card-soft",
        age=18,
        city="Москва",
        text_terms=("кофе/еда", "спокойный досуг"),
    )
    observation = MediaObservation(
        card_hash="card-soft",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:strong",
            "естественный/cute вайб",
            "очки",
            "домашняя обстановка",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert decision.score >= 3
    assert "positive_visual:face_match:strong" in decision.reasons
    assert "positive_visual:естественный/cute вайб" in decision.reasons


def test_autolike_decision_does_not_like_visible_face_without_face_match() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("1011",),
        text_hash="text",
        content_hash="card-soft-no-face-match",
        age=18,
        city="Москва",
        text_terms=("кофе/еда", "спокойный досуг"),
    )
    observation = MediaObservation(
        card_hash="card-soft-no-face-match",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "естественный/cute вайб",
            "очки",
            "домашняя обстановка",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "manual"
    assert "face_match_missing_for_visible_face" in decision.reasons


def test_autolike_decision_skips_visible_face_mismatch_even_with_body_and_text() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("1012",),
        text_hash="text",
        content_hash="card-anya-face-mismatch",
        age=18,
        city="Москва",
        text_terms=(
            "романтичность/родственная душа",
            "разносторонность/развитие",
            "кофе/еда",
        ),
    )
    observation = MediaObservation(
        card_hash="card-anya-face-mismatch",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:mismatch",
            "естественный/cute вайб",
            "естественная женственная фигура",
            "стройная/хрупкая фигура",
            "аккуратный женственный акцент",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "skip"
    assert decision.confidence >= 0.85
    assert decision.reasons == ("face_mismatch",)


def test_vision_tags_parse_face_not_taste_as_mismatch() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "лицо видно, но лицо не понравилось и не во вкус Никиты; "
        "черты лица не подходят под мягкий doll-like тип."
    )

    assert "face_match:mismatch" in tags


def test_autolike_decision_skips_full_face_or_large_body_even_if_cute() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("1013",),
        text_hash="text",
        content_hash="card-full-face-cute",
        age=18,
        city="Москва",
        text_terms=("спокойный досуг",),
    )
    observation = MediaObservation(
        card_hash="card-full-face-cute",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:weak",
            "естественный/cute вайб",
            "полное лицо",
            "крупная/полная фигура",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "skip"
    assert "hard_reject_visual_stop" in decision.reasons
    assert "крупная/полная фигура" in decision.reasons


def test_autolike_decision_skips_full_body_inferred_from_puffy_fingers() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("vika-puffy-fingers",),
        text_hash="text",
        content_hash="card-vika-puffy-fingers",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, natural/cute. Фигура выглядит "
            "полноватой; видны пухлые короткие пальцы-морковки и полная кисть."
        )
    )
    observation = MediaObservation(
        card_hash="card-vika-puffy-fingers",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "крупная/полная фигура" in tags
    assert decision.action == "skip"
    assert "крупная/полная фигура" in decision.reasons


def test_autolike_decision_skips_full_face_from_detached_chin_and_full_body() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("detached-chin-full-face",),
        text_hash="text",
        content_hash="card-detached-chin-full-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, closer_to_liked_cluster, natural/cute. "
            "face_detail: cheeks: soft/full, lower_third: soft, face_width: normal. "
            "Подбородок как будто отдельно от щек и визуально выпирает; "
            "body_frame: full, тело полное и подтверждает full-сигнал лица."
        )
    )
    observation = MediaObservation(
        card_hash="card-detached-chin-full-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "полное лицо" in tags
    assert "округло-пухловатое лицо" in tags
    assert decision.action == "skip"
    assert "полное лицо" in decision.reasons


def test_autolike_decision_skips_sagging_large_chest_even_if_face_weak() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("sofia-sagging-chest",),
        text_hash="text",
        content_hash="card-sofia-sagging-chest",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, сильный акцент на большой висячей "
            "груди, грудь выглядит обвисшей и является главным визуальным "
            "сигналом кадра."
        )
    )
    observation = MediaObservation(
        card_hash="card-sofia-sagging-chest",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "висячая большая грудь" in tags
    assert decision.action == "skip"
    assert "висячая большая грудь" in decision.reasons


def test_autolike_decision_likes_natural_city_photo_without_real_glam_stop() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("1014",),
        text_hash="text",
        content_hash="card-natural-city",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-natural-city",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:weak",
            "liked_cluster_face",
            "естественный/cute вайб",
            "городская обстановка",
            "эстетичный outfit",
            "стройная/хрупкая фигура",
            "гламур/студия",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert "hard_reject_visual_stop" not in decision.reasons
    assert "positive_visual:эстетичный outfit" in decision.reasons


def test_vision_tags_ignore_negated_glam_and_filter_stop_words() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "лицо видно, face_match:weak, городская обстановка, эстетичный outfit, "
        "не инстаграмный гламур, без пошлой сексуализации, без фильтров"
    )

    assert "face_match:weak" in tags
    assert "городская обстановка" in tags
    assert "эстетичный outfit" in tags
    assert "инстаграмный гламур" not in tags
    assert "пошлая сексуализация" not in tags
    assert "фильтр/маска" not in tags


def test_vision_tags_ignore_real_batch_negated_stop_word_contexts() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "лицо хорошо видно, мягкие черты, узкое/неполное лицо, "
        "повседневный кадр без явной гламурной или сексуализированной "
        "постановки, не модельная студийная подача, не выглядит полной"
    )

    assert "лицо видно" in tags
    assert "полное лицо" not in tags
    assert "крупная/полная фигура" not in tags
    assert "гламур/студия" not in tags
    assert "искусственная гламурная подача" not in tags
    assert "пошлая сексуализация" not in tags


def test_vision_tags_do_not_infer_full_face_from_negated_chin_signal() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "лицо видно, face_match:weak, closer_to_liked_cluster, "
        "quality_limited_face_match. face_detail: lips: large, "
        "lip_expression: natural, brows: neat/thin, cheeks: slim, "
        "lower_third: thin, face_width: narrow. Подбородок не выпирает "
        "и не отделен от щек; лицо не полное."
    )

    assert "полное лицо" not in tags
    assert "округло-пухловатое лицо" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_stop_evidence_absence_phrasing() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match: weak. Черты естественные, губы без гламурной/филлерной "
        "подачи. stop_evidence: явных признаков пошлости, инстаграмного "
        "гламура/филлеров или heavy filter не видно."
    )

    assert "face_match:weak" in tags
    assert "гламур/студия" not in tags
    assert "инстаграмный гламур" not in tags
    assert "искусственная гламурная подача" not in tags
    assert "накачанные губы/филлеры" not in tags
    assert "пошлая сексуализация" not in tags
    assert "сильный фильтр/маска" not in tags


def test_vision_tags_ignore_negated_glam_model_face_phrasing() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster. Natural/casual, без гламура "
        "и сексуализации, не модельная подача."
    )

    assert "face_match:weak" in tags
    assert "гламурно-модельное лицо" not in tags
    assert "гламур/студия" not in tags


def test_vision_tags_liked_quality_limited_face_overrides_soft_cold_note() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
            "Лицо хорошо видно, узкое/мягкое, expressive eyes. "
            "Компактная liked-геометрия лица: компактная центральная зона, "
            "большие открытые округлые глаза, короткий аккуратный нос. "
            "stop_evidence: холодное/нейтральное лицо."
        )
    )
    card = TasteCard(
        message_ids=("liked-cold-note",),
        text_hash="text",
        content_hash="card-liked-cold-note",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-liked-cold-note",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "liked_cluster_face" in tags
    assert "компактная liked-геометрия лица" in tags
    assert "холодное/нейтральное лицо" not in tags
    assert "face_match:mismatch" not in tags
    assert decision.action == "like"


def test_autolike_decision_skips_uncertain_cute_cold_face_even_if_liked_cluster() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster. face_shape:slim / soft. "
            "face_detail: lips: normal, brows: neat, cheeks: slim, "
            "lower_third: soft/thin, face_width: narrow/normal. "
            "stop_evidence: uncertain_cute_face, холодное/нейтральное лицо, "
            "сухое нейтральное выражение."
        )
    )
    card = TasteCard(
        message_ids=("uncertain-cold-liked-cluster",),
        text_hash="text",
        content_hash="card-uncertain-cold-liked-cluster",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-uncertain-cold-liked-cluster",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "liked_cluster_face" in tags
    assert "uncertain_cute_face" in tags
    assert "холодное/нейтральное лицо" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "холодное/нейтральное лицо" in decision.reasons


def test_vision_tags_rescue_classic_slim_harmony_from_soft_mismatch() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "face_match:mismatch, closer_to_disliked_cluster, disliked_cluster_face. "
            "face_shape: slim/soft, но не liked-кластер. "
            "face_detail: lips: normal; lip_expression: relaxed/natural; "
            "brows: normal; cheeks: slim/soft; lower_third: soft; "
            "face_width: normal. stop_evidence: холодное/нейтральное лицо, "
            "недостаточно doll-like лицо, не похоже на liked-кластер."
        )
    )
    card = TasteCard(
        message_ids=("classic-slim-soft-mismatch",),
        text_hash="text",
        content_hash="card-classic-slim-soft-mismatch",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-classic-slim-soft-mismatch",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "face_match:weak" in tags
    assert "liked_cluster_face" in tags
    assert "классическая slim-гармония лица" in tags
    assert "disliked_cluster_face" not in tags
    assert "face_match:mismatch" not in tags
    assert decision.action == "like"


def test_vision_tags_do_not_rescue_explicit_disliked_face_reference() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "reference_classification\n"
            "face_reference_class: disliked_face\n"
            "reference_confidence: 0.97\n"
            "face_match:mismatch, closer_to_disliked_cluster, disliked_cluster_face, "
            "не похоже на liked-кластер, недостаточно doll-like лицо. "
            "face_detail: lips: normal; lip_expression: natural/relaxed; "
            "brows: neat/normal; cheeks: soft/slim; lower_third: soft; "
            "face_width: normal."
        )
    )
    card = TasteCard(
        message_ids=("explicit-disliked-reference",),
        text_hash="text",
        content_hash="card-explicit-disliked-reference",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-explicit-disliked-reference",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "disliked_cluster_face" in tags
    assert "liked_cluster_face" not in tags
    assert "классическая slim-гармония лица" not in tags
    assert decision.action == "skip"


def test_vision_tags_does_not_rescue_noncompact_soft_mismatch() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "face_match:mismatch, closer_to_disliked_cluster, disliked_cluster_face. "
            "face_detail: lips: normal; lip_expression: relaxed/natural; "
            "brows: normal; cheeks: slim/soft; lower_third: soft; "
            "face_width: normal. Средняя треть лица некомпактная: "
            "eye-to-mouth zone longer, eyes narrow/elongated, nose longer. "
            "stop_evidence: холодное/нейтральное лицо, не похоже на liked-кластер."
        )
    )
    card = TasteCard(
        message_ids=("noncompact-soft-mismatch",),
        text_hash="text",
        content_hash="card-noncompact-soft-mismatch",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-noncompact-soft-mismatch",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "некомпактная средняя треть лица" in tags
    assert "disliked_cluster_face" in tags
    assert decision.action == "skip"


def test_vision_tags_liked_cluster_pout_note_does_not_force_glam_stop() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
        "Мягкие черты, очки, челка, natural/quirky; ракурс и выражение с pout "
        "дают не максимальную уверенность для strong."
    )

    assert "liked_cluster_face" in tags
    assert "гламурно-модельное лицо" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_absent_stop_words_from_real_batch_phrasing() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:strong, body_frame:slim. stop_evidence: невыгодное "
        "искажение лица не вижу; полной кисти/пухлых пальцев не видно; "
        "висячая большая грудь не подтверждается; гламурной или "
        "сексуализированной подаче не вижу."
    )

    assert "невыгодное искажение лица" not in tags
    assert "крупная/полная фигура" not in tags
    assert "висячая большая грудь" not in tags
    assert "гламур/студия" not in tags
    assert "пошлая сексуализация" not in tags


def test_vision_tags_ignore_russian_negated_heavy_filter_phrase() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster, natural/cute, "
        "без явного heavy filter/glam/сексуализации."
    )

    assert "face_match:weak" in tags
    assert "сильный фильтр/маска" not in tags
    assert "гламур/студия" not in tags
    assert "пошлая сексуализация" not in tags


def test_vision_tags_treat_not_confident_enough_face_as_weak_not_mismatch() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:mismatch. Фото аккуратное и естественное, мягкий портрет, "
        "но само лицо не попадает достаточно уверенно в мягкий хрупкий "
        "doll-like типаж."
    )

    assert "face_match:mismatch" not in tags
    assert "face_match:weak" in tags


def test_vision_tags_treat_blurry_overexposed_face_mismatch_as_weak() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:mismatch. Лицо видно, но кадр размытый и сильно "
        "пересвеченный; мягкий хрупкий doll-like типаж выражен слабо."
    )

    assert "face_match:mismatch" not in tags
    assert "face_match:weak" in tags


def test_vision_tags_ignore_sagging_chest_when_it_is_not_an_accent() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, body_frame:slim. Большая висячая грудь не является "
        "акцентом кадра."
    )

    assert "висячая большая грудь" not in tags


def test_vision_tags_parse_visible_not_taste_face_as_mismatch() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "лицо видно, зимний естественный кадр, лицо приятное, но не попадает "
        "во вкус Никиты и не тот мягкий doll-like тип"
    )

    assert "face_match:mismatch" in tags
    assert "лицо видно" in tags


def test_vision_tags_parse_markdown_face_match_value() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match: **weak**\nface_shape: **soft**\n"
        "presentation: hoodie/casual, natural, cute, quirky"
    )

    assert "face_match:weak" in tags


def test_vision_tags_parse_backticked_face_match_value() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match: `weak`, quality_limited_face_match, closer_to_liked_cluster"
    )

    assert "face_match:weak" in tags
    assert "liked_cluster_face" in tags


def test_vision_tags_ignore_inverted_negated_instagram_glam_phrase() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster. stop_evidence: не видно "
        "округло-пухлого лица, массивной нижней трети, доминирующих крупных "
        "губ, тяжелой связки губ-бровей-щек, инстаграмного гламура или "
        "сексуализированной подачи."
    )

    assert "face_match:weak" in tags
    assert "инстаграмный гламур" not in tags
    assert "гламур/студия" not in tags


def test_vision_tags_ignore_english_negated_wide_and_not_liked_stops() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:strong, closer_to_liked_cluster. face_shape: slim / soft. "
        "face_detail: lips: normal-large; lip_expression: natural/relaxed; "
        "brows: neat/normal; cheeks: slim/soft; lower_third: thin/soft; "
        "face_width: narrow-normal. stop_evidence: значимых disliked-stop "
        "признаков по лицу не видно; нет full/wide/heavy lower third, "
        "нет glam/filler/pout-сигнала, нет `не похоже на liked-кластер`."
    )

    assert "face_match:strong" in tags
    assert "широкое/массивное лицо" not in tags
    assert "массивная нижняя треть и огромные губы" not in tags
    assert "искусственная гламурная подача" not in tags
    assert "не похоже на liked-кластер" not in tags


def test_vision_tags_ignore_without_glam_and_without_wide_face_examples() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
        "Сравнение с reference sheets: candidate ближе к liked_face: "
        "узкое/нормальное лицо, мягкая нижняя треть, без rejected-сигналов "
        "вроде full/wide face, pout/glam или тяжелой связки губ-бровей-щек. "
        "Лицо мягкое/slim, без glam/pout/filler-сигнала."
    )

    assert "широкое/массивное лицо" not in tags
    assert "искусственная гламурная подача" not in tags
    assert "гламур/студия" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_visible_stop_names_listed_as_absent() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:strong, closer_to_liked_cluster. face_shape: slim / soft. "
        "stop_evidence: видимых признаков `полное лицо`, "
        "`широкое/массивное лицо`, `доминирующие крупные губы`, `грубое лицо`, "
        "`инстаграмный гламур/филлеры` или body-stop по фигуре нет."
    )

    assert "face_match:strong" in tags
    assert "полное лицо" not in tags
    assert "широкое/массивное лицо" not in tags
    assert "грубое лицо" not in tags
    assert "инстаграмный гламур" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_long_absent_glam_list() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:strong, closer_to_liked_cluster. "
        "С disliked_face совпадение слабое: нет округло-пухлого лица, тяжелой "
        "связки губ-бровей-щек, доминирующих губ, инстаграмной/модельной "
        "подачи или грубого выражения. stop_evidence: нет крупной фигуры, "
        "висячей груди, glam/филлеров, pout или широкого лица."
    )

    assert "инстаграмный гламур" not in tags
    assert "искусственная гламурная подача" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_absence_of_full_wide_face_phrase() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
        "От rejected face sheets отличается отсутствием full/wide face, "
        "heavy brows, dominant pout-губ, glam/model подачи."
    )

    assert "широкое/массивное лицо" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_absence_of_full_round_and_heavy_lower_third() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:strong, closer_to_liked_cluster. "
        "От disliked_face отличается отсутствием full/round лица, "
        "heavy lower third, pout/glam-филлерного акцента."
    )

    assert "округло-пухловатое лицо" not in tags
    assert "широкое/массивное лицо" not in tags
    assert "массивная нижняя треть и огромные губы" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_ignore_russian_negated_heavy_lower_third() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
        "Лицо ближе к liked_face: узко-нормальная ширина, мягкие щеки, "
        "не тяжелая нижняя треть, естественные губы."
    )

    assert "широкое/массивное лицо" not in tags
    assert "массивная нижняя треть и огромные губы" not in tags
    assert "face_match:mismatch" not in tags


def test_vision_tags_keep_liked_compatible_large_lips_with_soft_lower_third() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
        "face_detail: lips: large; lip_expression: pout; brows: neat; "
        "cheeks: slim; lower_third: soft; face_width: narrow. "
        "stop_evidence: губы визуально крупные, но без тяжелых бровей, "
        "округлых щек или массивной нижней трети."
    )

    assert "liked_cluster_face" in tags
    assert "доминирующие крупные губы" not in tags
    assert "face_match:mismatch" not in tags


def test_autolike_decision_skips_unknown_face_without_body_fit() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("landscape-only",),
        text_hash="text",
        content_hash="card-landscape-only",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-landscape-only",
        media_hash="m1",
        kind="photo",
        tags=(
            "face_match:unknown",
            "естественный/cute вайб",
            "городская обстановка",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "skip"
    assert "no_visible_face_or_body_fit" in decision.reasons


def test_autolike_decision_routes_age_risk_to_manual_review() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("102",),
        text_hash="text",
        content_hash="card-age-risk",
        age=17,
        text_terms=("кофе/еда",),
    )
    observation = MediaObservation(
        card_hash="card-age-risk",
        media_hash="m1",
        kind="photo",
        tags=("лицо видно", "естественный/cute вайб"),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "manual"
    assert "manual_age_review" in decision.reasons


def test_autolike_decision_prefers_moscow_and_age_18_to_20_softly() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    observation = MediaObservation(
        card_hash="card",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:strong",
            "естественный/cute вайб",
            "очки",
            "домашняя обстановка",
        ),
    )

    baseline = TasteCard(
        message_ids=("201",),
        text_hash="text",
        content_hash="card",
        age=21,
        city="Казань",
        text_terms=("кофе/еда",),
    )
    msk_ok = baseline.model_copy(update={"age": 20, "city": "Msk"})
    under18 = baseline.model_copy(update={"age": 17, "city": "Москва"})

    baseline_decision = decide_daivinchik_autolike(
        baseline, observations=(observation,)
    )
    msk_decision = decide_daivinchik_autolike(msk_ok, observations=(observation,))
    under18_decision = decide_daivinchik_autolike(under18, observations=(observation,))

    assert baseline_decision.action != "skip"
    assert "preferred_age:18-20" not in baseline_decision.reasons
    assert "preferred_city:moscow" not in baseline_decision.reasons
    assert msk_decision.action == "like"
    assert "preferred_age:18-20" in msk_decision.reasons
    assert "preferred_city:moscow" in msk_decision.reasons
    assert under18_decision.action == "manual"
    assert under18_decision.reasons == ("manual_age_review",)


def test_autolike_decision_likes_user_corrected_natural_redhead_freckles() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("202",),
        text_hash="text",
        content_hash="card-natural-redhead",
        age=19,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-natural-redhead",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:strong",
            "естественный/cute вайб",
            "рыжие/медные волосы",
            "веснушки",
            "уютный casual стиль",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert decision.score >= 5
    assert "positive_visual_correction:natural_ginger_freckles" in decision.reasons


def test_autolike_decision_hard_skips_negative_visual_presentation() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("103",),
        text_hash="text",
        content_hash="card-body-first",
        age=19,
        text_terms=("игры/онлайн",),
    )
    observation = MediaObservation(
        card_hash="card-body-first",
        media_hash="m1",
        kind="photo",
        tags=(
            "body-first/mirror-first",
            "инстаграмный гламур",
            "искусственная гламурная подача",
            "накачанные губы/филлеры",
            "пошлая сексуализация",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "skip"
    assert "hard_reject_visual_stop" in decision.reasons


def test_autolike_decision_skips_distorted_closeup_even_if_weak_cute_match() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("nastya-closeup",),
        text_hash="text",
        content_hash="card-nastya-closeup",
        age=18,
        city="Москва",
        text_terms=("игры/онлайн", "фандом/аниме"),
    )
    observation = MediaObservation(
        card_hash="card-nastya-closeup",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:weak",
            "очки",
            "челка",
            "естественный/cute вайб",
            "quirky/nerdy вайб",
            "невыгодное искажение лица",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "skip"
    assert "hard_reject_visual_stop" in decision.reasons
    assert "невыгодное искажение лица" in decision.reasons


def test_autolike_decision_allows_mild_closeup_distortion_when_face_matches() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("liked-closeup",),
        text_hash="text",
        content_hash="card-liked-closeup",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak. Лицо читается хорошо: мягкие черты, выразительные "
            "глаза. Ракурс сверху и близкий selfie немного искажает пропорции, "
            "поэтому не strong. stop_evidence: явных стоп-признаков не видно."
        )
    )
    observation = MediaObservation(
        card_hash="card-liked-closeup",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "невыгодное искажение лица" not in tags
    assert decision.action == "like"


def test_autolike_decision_allows_quality_limited_weak_face_match() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("liked-quality-limited",),
        text_hash="text",
        content_hash="card-liked-quality-limited",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak. Лицо близко к мягкому хрупкому doll-like типу: "
            "выразительные глаза, легкая нижняя треть, естественные губы. "
            "Но кадр размытый, лицо частично закрыто, поэтому это "
            "quality_limited_match, не strong."
        )
    )
    observation = MediaObservation(
        card_hash="card-liked-quality-limited",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "quality_limited_face_match" in tags
    assert decision.action == "like"


def test_autolike_decision_routes_uncertain_cute_face_to_manual() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("uncertain-small-face",),
        text_hash="text",
        content_hash="card-uncertain-small-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
            "Кадр не крупный, лицо читается ограниченно; выражение немного "
            "прищуренное и не дает уверенного милого/cute сигнала, но hard-stop "
            "по форме лица, губам, фигуре или гламуру не видно."
        )
    )
    observation = MediaObservation(
        card_hash="card-uncertain-small-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "uncertain_cute_face" in tags
    assert "лицо мелкое/далеко" in tags
    assert "face_match:mismatch" not in tags
    assert decision.action == "manual"
    assert "uncertain_cute_face" in decision.reasons


def test_autolike_decision_does_not_manual_uncertain_cute_when_face_is_close() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("uncertain-close-face",),
        text_hash="text",
        content_hash="card-uncertain-close-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
            "Лицо хорошо видно, крупный план, soft/slim, но stop_evidence: "
            "uncertain_cute_face из-за мягкого качества кадра. Компактная "
            "liked-геометрия лица: компактная центральная зона, большие "
            "открытые округлые глаза, короткий аккуратный нос."
        )
    )
    observation = MediaObservation(
        card_hash="card-uncertain-close-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "uncertain_cute_face" in tags
    assert "лицо мелкое/далеко" not in tags
    assert decision.action == "like"


def test_autolike_decision_allows_uncertain_liked_cluster_when_body_fit_rescues() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("uncertain-small-face-fit",),
        text_hash="text",
        content_hash="card-uncertain-small-face-fit",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
            "Кадр не крупный, лицо читается ограниченно, но общий образ подходит: "
            "стройная/хрупкая фигура, эстетичный outfit. "
            "Компактная liked-геометрия лица: компактная центральная зона, "
            "мягкая линия нижней челюсти."
        )
    )
    observation = MediaObservation(
        card_hash="card-uncertain-small-face-fit",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "лицо мелкое/далеко" in tags
    assert "liked_cluster_face" in tags
    assert "стройная/хрупкая фигура" in tags
    assert decision.action == "like"


def test_autolike_decision_skips_far_generic_soft_face_even_with_body_fit() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("far-generic-soft-face-fit",),
        text_hash="text",
        content_hash="card-far-generic-soft-face-fit",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
            "Лицо мелкое/далеко, uncertain_cute_face. face_detail: lips: normal; "
            "lip_expression: relaxed/natural; brows: neat/normal; cheeks: soft; "
            "lower_third: soft; face_width: normal. body_frame: slim/average, "
            "стройная/хрупкая фигура."
        )
    )
    observation = MediaObservation(
        card_hash="card-far-generic-soft-face-fit",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "лицо мелкое/далеко" in tags
    assert "классическая slim-гармония лица" not in tags
    assert decision.action == "skip"


def test_autolike_decision_allows_closed_face_when_body_fit_rescues() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("closed-face-body-fit",),
        text_hash="text",
        content_hash="card-closed-face-body-fit",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
            "Лицо закрыто телефоном, но видимая часть ближе к liked. "
            "body_frame: slim/petite, стройная/хрупкая фигура, "
            "естественная женственная фигура, liked_body."
        )
    )
    observation = MediaObservation(
        card_hash="card-closed-face-body-fit",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "лицо закрыто" in tags
    assert "liked_cluster_face" in tags
    assert "стройная/хрупкая фигура" in tags
    assert decision.action == "like"


def test_autolike_decision_rejects_large_lips_without_slim_cheeks() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("large-lips-soft-lower-third",),
        text_hash="text",
        content_hash="card-large-lips-soft-lower-third",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
            "face_detail:\n"
            "- lips: large\n"
            "- lip_expression: natural/pout\n"
            "- brows: neat\n"
            "- cheeks: soft\n"
            "- lower_third: soft\n"
            "- face_width: normal/narrow\n"
            "presentation: natural/cute, стройная/хрупкая фигура."
        )
    )
    observation = MediaObservation(
        card_hash="card-large-lips-soft-lower-third",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "доминирующие крупные губы" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"


def test_autolike_decision_rejects_filtered_generic_soft_pout_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("filtered-generic-soft-pout",),
        text_hash="text",
        content_hash="card-filtered-generic-soft-pout",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:strong, closer_to_liked_cluster. "
            "face_detail:\n"
            "- lips: normal\n"
            "- lip_expression: pout\n"
            "- brows: neat\n"
            "- cheeks: soft\n"
            "- lower_third: soft\n"
            "- face_width: normal\n"
            "presentation: natural/cute, фильтры, домашняя обстановка."
        )
    )
    observation = MediaObservation(
        card_hash="card-filtered-generic-soft-pout",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "не похоже на liked-кластер" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"


def test_autolike_decision_keeps_soft_pout_when_liked_geometry_is_supported() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("liked-soft-pout",),
        text_hash="text",
        content_hash="card-liked-soft-pout",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
            "face_detail:\n"
            "- lips: normal/large\n"
            "- lip_expression: pout\n"
            "- brows: neat\n"
            "- cheeks: slim\n"
            "- lower_third: soft\n"
            "- face_width: narrow\n"
            "presentation: natural/cute, стройная/хрупкая фигура. "
            "Компактная liked-геометрия лица: компактная центральная зона, "
            "большие открытые округлые глаза, короткий аккуратный нос."
        )
    )
    observation = MediaObservation(
        card_hash="card-liked-soft-pout",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "доминирующие крупные губы" not in tags
    assert "не похоже на liked-кластер" not in tags
    assert "компактная liked-геометрия лица" in tags
    assert decision.action == "like"


def test_autolike_decision_skips_quality_limited_video_face_mismatch() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("liked-video-frame",),
        text_hash="text",
        content_hash="card-liked-video-frame",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:mismatch, closer_to_disliked_cluster. Кадр размытый и "
            "пересвеченный, quality_limited_face_visibility; natural/cute, "
            "стройная/хрупкая фигура. face_detail: lips: large, "
            "lip_expression: dominant, cheeks: slim, lower_third: soft."
        )
    )
    observation = MediaObservation(
        card_hash="card-liked-video-frame",
        media_hash="m1",
        kind="video",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "quality_limited_face_match" in tags
    assert "доминирующие крупные губы" in tags
    assert decision.action == "skip"
    assert "face_mismatch" in decision.reasons


def test_autolike_decision_skips_quality_limited_mismatch_even_with_positive_visuals() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("quality-limited-positive-visual",),
        text_hash="text",
        content_hash="card-quality-limited-positive-visual",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:mismatch, closer_to_disliked_cluster, "
            "quality_limited_face_match, не похоже на liked-кластер. "
            "Лицо generic soft close-up, но face_detail без hard-stop: "
            "lips: normal, brows: neat, cheeks: slim, lower_third: soft. "
            "Natural/cute, эстетичный outfit, естественная женственная фигура, "
            "стройная/хрупкая фигура."
        )
    )
    observation = MediaObservation(
        card_hash="card-quality-limited-positive-visual",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "quality_limited_face_match" in tags
    assert "disliked_cluster_face" in tags
    assert "не похоже на liked-кластер" in tags
    assert decision.action == "skip"
    assert "не похоже на liked-кластер" in decision.reasons


def test_autolike_decision_routes_weak_visible_face_without_liked_cluster_to_manual() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("weak-visible-generic-face",),
        text_hash="text",
        content_hash="card-weak-visible-generic-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, quality_limited_face_match. "
            "Лицо soft/natural/cute, очки и уютный капюшон, но явного "
            "сходства с понравившимся кластером нет. Эстетичный outfit, "
            "естественная женственная фигура, "
            "стройная/хрупкая фигура."
        )
    )
    observation = MediaObservation(
        card_hash="card-weak-visible-generic-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "face_match:weak" in tags
    assert "liked_cluster_face" not in tags
    assert decision.action == "manual"
    assert decision.reasons == ("weak_face_without_liked_cluster",)


def test_autolike_decision_likes_liked_cluster_even_with_tense_expression() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("liked-tense-face",),
        text_hash="text",
        content_hash="card-liked-tense-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, closer_to_liked_cluster, "
            "quality_limited_face_match, uncertain_cute_face, "
            "напряженное/прищуренное лицо. face_detail: lips: normal, "
            "brows: neat, cheeks: slim, lower_third: thin. Natural/cute, "
            "стройная/хрупкая фигура. Компактная liked-геометрия лица: "
            "компактная центральная зона, большие открытые округлые глаза, "
            "короткий аккуратный нос."
        )
    )
    observation = MediaObservation(
        card_hash="card-liked-tense-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "liked_cluster_face" in tags
    assert "компактная liked-геометрия лица" in tags
    assert decision.action == "like"


def test_vision_tags_do_not_treat_artificial_light_as_glam_stop() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
            "Качество кадра: теплый искусственный свет, natural, slightly quirky."
        )
    )

    assert "искусственная гламурная подача" not in tags


def test_autolike_decision_skips_weak_face_when_not_doll_like_enough() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("disliked-natural-cute-not-doll",),
        text_hash="text",
        content_hash="card-disliked-natural-cute-not-doll",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak. Лицо мягкое natural/cute, но не хватает "
            "хрупкой doll-like геометрии; лицо скорее natural-pretty, чем "
            "кукольное, выражение нейтральное и немного холодное."
        )
    )
    observation = MediaObservation(
        card_hash="card-disliked-natural-cute-not-doll",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "недостаточно doll-like лицо" in tags
    assert "холодное/нейтральное лицо" in tags
    assert decision.action == "skip"
    assert "недостаточно doll-like лицо" in decision.reasons


def test_autolike_decision_skips_slim_but_rough_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("rough-slim-face",),
        text_hash="text",
        content_hash="card-rough-slim-face",
        age=18,
        city="Москва",
        text_terms=("творчество/искусство", "романтичность/родственная душа"),
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, face_shape:slim, body_frame:slim. "
            "Лицо худое и не полное, но впечатление harsh/coarse: тяжелые "
            "брови, hard eye area, резкий напряженный взгляд, грубые черты "
            "и недостаточно деликатное лицо."
        )
    )
    observation = MediaObservation(
        card_hash="card-rough-slim-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "грубое лицо" in tags
    assert "face_match:mismatch" in tags
    assert "полное лицо" not in tags
    assert decision.action == "skip"
    assert "грубое лицо" in decision.reasons


def test_autolike_decision_skips_round_puffy_face_even_if_natural_cute() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("round-puffy-face",),
        text_hash="text",
        content_hash="card-round-puffy-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, natural/cute, но лицо округло-пухловатое: "
            "мягко-округлое лицо, пухловатые щеки и не очень тонкая нижняя треть."
        )
    )
    observation = MediaObservation(
        card_hash="card-round-puffy-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "округло-пухловатое лицо" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "округло-пухловатое лицо" in decision.reasons


def test_autolike_decision_skips_massive_lower_third_with_huge_lips() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("massive-lower-third-huge-lips",),
        text_hash="text",
        content_hash="card-massive-lower-third-huge-lips",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, но широкое массивное лицо: массивная "
            "нижняя треть и огромные губы, нижняя часть лица выглядит тяжелой."
        )
    )
    observation = MediaObservation(
        card_hash="card-massive-lower-third-huge-lips",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "широкое/массивное лицо" in tags
    assert "массивная нижняя треть и огромные губы" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "массивная нижняя треть и огромные губы" in decision.reasons


def test_autolike_decision_skips_large_wide_flat_facial_parts() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("large-wide-flat-face-parts",),
        text_hash="text",
        content_hash="card-large-wide-flat-face-parts",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, natural/cute, но rejected-сигнал: "
            "крупные части лица, крупный нос, широкая челюсть, широкие скулы "
            "и плосковатое широкое впечатление."
        )
    )
    observation = MediaObservation(
        card_hash="card-large-wide-flat-face-parts",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "широкое/массивное лицо" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "широкое/массивное лицо" in decision.reasons


def test_autolike_decision_skips_non_compact_midface_balance() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("non-compact-midface",),
        text_hash="text",
        content_hash="card-non-compact-midface",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, natural/cute. "
            "Но средняя часть лица кажется длиннее: зона от глаз до губ "
            "визуально длиннее, глаза выглядят более узкими и есть тяжелое "
            "верхнее веко; губы выглядят темнее и напряженно сжатыми."
        )
    )
    observation = MediaObservation(
        card_hash="card-non-compact-midface",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "некомпактная средняя треть лица" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "некомпактная средняя треть лица" in decision.reasons


def test_autolike_decision_skips_weak_quality_limited_without_compact_geometry() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("weak-no-compact",),
        text_hash="text",
        content_hash="card-weak-no-compact",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, "
            "quality_limited_face_match. Natural/cute, slim/soft, но без "
            "явного описания компактной liked-геометрии."
        )
    )
    observation = MediaObservation(
        card_hash="card-weak-no-compact",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "компактная liked-геометрия лица" not in tags
    assert decision.action == "skip"
    assert decision.reasons == ("weak_quality_limited_without_compact_liked_geometry",)


def test_autolike_decision_allows_weak_quality_limited_with_compact_geometry() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("weak-compact",),
        text_hash="text",
        content_hash="card-weak-compact",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, "
            "quality_limited_face_match. Компактная liked-геометрия лица: "
            "компактная центральная зона, большие открытые округлые глаза, "
            "короткий аккуратный нос, мягкая линия нижней челюсти, "
            "гармоничное соотношение глаз, носа и губ."
        )
    )
    observation = MediaObservation(
        card_hash="card-weak-compact",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "компактная liked-геометрия лица" in tags
    assert decision.action == "like"


def test_autolike_decision_allows_weak_quality_limited_with_classic_slim_harmony() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("weak-classic-slim-harmony",),
        text_hash="text",
        content_hash="card-weak-classic-slim-harmony",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, "
            "quality_limited_face_match. Классическая slim-гармония лица: "
            "нормальные губы, аккуратные брови, soft cheeks, soft lower_third, "
            "normal/narrow face_width. Natural/cute, стройная/хрупкая фигура."
        )
    )
    observation = MediaObservation(
        card_hash="card-weak-classic-slim-harmony",
        media_hash="m1",
        kind="video",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "классическая slim-гармония лица" in tags
    assert decision.action == "like"


def test_autolike_decision_derives_classic_slim_harmony_from_face_detail() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("weak-derived-classic-slim",),
        text_hash="text",
        content_hash="card-weak-derived-classic-slim",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match: weak, quality_limited_face_match, closer_to_liked_cluster. "
            "face_detail: lips: normal; lip_expression: relaxed/natural; "
            "brows: neat/normal; cheeks: soft/slim; lower_third: soft/thin; "
            "face_width: normal/narrow. stop_evidence: quality_limited_face_match."
        )
    )
    observation = MediaObservation(
        card_hash="card-weak-derived-classic-slim",
        media_hash="m1",
        kind="video",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "классическая slim-гармония лица" in tags
    assert decision.action == "like"


def test_autolike_decision_derives_classic_slim_harmony_with_playful_expression() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("weak-playful-classic-slim",),
        text_hash="text",
        content_hash="card-weak-playful-classic-slim",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match: weak, quality_limited_face_match, closer_to_liked_cluster. "
            "face_detail: lips: normal; lip_expression: playful/natural; "
            "brows: normal/neat; cheeks: slim/soft; lower_third: soft; "
            "face_width: normal/narrow."
        )
    )
    observation = MediaObservation(
        card_hash="card-weak-playful-classic-slim",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "классическая slim-гармония лица" in tags
    assert decision.action == "like"


def test_autolike_decision_skips_heavy_lips_brows_cheeks_combo() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("heavy-lips-brows-cheeks",),
        text_hash="text",
        content_hash="card-heavy-lips-brows-cheeks",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, natural/cute, но крупные губы, "
            "густые брови и округлые щеки вместе дают тяжелую связку "
            "губ-бровей-щек."
        )
    )
    observation = MediaObservation(
        card_hash="card-heavy-lips-brows-cheeks",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "тяжелая связка губ-бровей-щек" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "тяжелая связка губ-бровей-щек" in decision.reasons


def test_autolike_decision_skips_full_heavy_lower_third_even_with_cute_vibe() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("full-heavy-lower-third",),
        text_hash="text",
        content_hash="card-full-heavy-lower-third",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, natural/cute, но в 3/4 нижняя треть "
            "выглядит full/heavy: щека челюсть подбородок дают цельный "
            "округлый объем, нет сужения к подбородку, короткий/мягкий "
            "подбородок, низ лица главный визуальный вес."
        )
    )
    observation = MediaObservation(
        card_hash="card-full-heavy-lower-third",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "нехрупкая нижняя треть лица" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "нехрупкая нижняя треть лица" in decision.reasons


def test_autolike_decision_skips_soft_full_face_misread_as_slim() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("soft-full-not-slim-delicate",),
        text_hash="text",
        content_hash="card-soft-full-not-slim-delicate",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, natural/cute, но лицо "
            "не даёт ощущения тонкого/slim силуэта: оно воспринимается "
            "широковатым и плотным, средняя часть лица выглядит полной, "
            "soft-full лицо, мягкость воспринимается как fullness. Нижняя треть "
            "не даёт slim/V-line impression и нет красивого тонкого сужения."
        )
    )
    observation = MediaObservation(
        card_hash="card-soft-full-not-slim-delicate",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "полное лицо" in tags
    assert "широкое/массивное лицо" in tags
    assert "нехрупкая нижняя треть лица" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"


def test_vision_tags_do_not_treat_face_slim_as_body_fit() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
        "face_shape:soft/slim. body_frame:unknown. face_detail: lips: normal; "
        "brows: neat; cheeks: slim; lower_third: soft; face_width: normal/narrow."
    )

    assert "стройная/хрупкая фигура" not in tags


def test_vision_tags_do_not_treat_feminine_style_accent_as_body_fit() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
        "body_frame:unknown. presentation: casual mirror selfie, natural/cute, "
        "аккуратный женственный акцент, фигура почти скрыта oversized hoodie."
    )

    assert "естественная женственная фигура" not in tags


def test_autolike_decision_skips_cold_tense_quality_limited_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("cold-tense-quality-limited",),
        text_hash="text",
        content_hash="card-cold-tense-quality-limited",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
            "face_shape:soft/slim. body_frame:unknown. face_detail: lips: normal; "
            "lip_expression: natural; brows: neat/normal; cheeks: soft/slim; "
            "lower_third: soft; face_width: normal/narrow. stop_evidence: "
            "холодное/нейтральное лицо, слегка напряженное выражение, не strong "
            "из-за ракурса."
        )
    )
    observation = MediaObservation(
        card_hash="card-cold-tense-quality-limited",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "холодное/нейтральное лицо" in tags
    assert "напряженное/прищуренное лицо" in tags
    assert "стройная/хрупкая фигура" not in tags
    assert "классическая slim-гармония лица" not in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"


def test_autolike_decision_skips_partly_covered_cold_quality_limited_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("covered-cold-quality-limited",),
        text_hash="text",
        content_hash="card-covered-cold-quality-limited",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match: weak, closer_to_liked_cluster, quality_limited_face_match. "
            "face_shape: slim/soft. body_frame: slim-to-average. face_detail: "
            "lips: normal, brows: neat, cheeks: slim/soft, lower_third: soft, "
            "face_width: normal/narrow. presentation: лицо видно частично "
            "закрыто телефоном, эстетичный outfit. stop_evidence: выражение "
            "нейтральное/слегка холодное, поэтому не strong."
        )
    )
    observation = MediaObservation(
        card_hash="card-covered-cold-quality-limited",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "холодное/нейтральное лицо" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"


def test_autolike_decision_skips_partly_covered_genitive_cold_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("covered-genitive-cold-face",),
        text_hash="text",
        content_hash="card-covered-genitive-cold-face",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster, quality_limited_face_match. "
            "face_shape:slim / soft. body_frame:slim. face_detail: lips: normal; "
            "brows: neat; cheeks: slim/soft; lower_third: soft; face_width: "
            "normal/narrow. presentation: лицо частично перекрыто телефоном, "
            "эстетичный outfit. stop_evidence: не strong из-за частичного "
            "закрытия лица телефоном и спокойного/слегка холодного выражения."
        )
    )
    observation = MediaObservation(
        card_hash="card-covered-genitive-cold-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "холодное/нейтральное лицо" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"


def test_autolike_decision_manual_for_uncertain_dark_liked_cluster_without_fit() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("uncertain-dark-liked-cluster",),
        text_hash="text",
        content_hash="card-uncertain-dark-liked-cluster",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, quality_limited_face_match, closer_to_liked_cluster. "
            "face_shape:soft/slim, лицо читается не полностью из-за темноты и "
            "шума. body_frame:unknown. face_detail: lips: normal; "
            "lip_expression: relaxed/natural; brows: normal/unclear; "
            "cheeks: slim/soft; lower_third: soft; face_width: narrow/normal. "
            "stop_evidence: качество кадра, видно ли лицо: лицо видно, но "
            "темное и шумное; uncertain_cute_face."
        )
    )
    observation = MediaObservation(
        card_hash="card-uncertain-dark-liked-cluster",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "uncertain_cute_face" in tags
    assert "компактная liked-геометрия лица" not in tags
    assert decision.action == "manual"


def test_autolike_decision_skips_large_lips_without_fragile_geometry() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("large-lips-soft-cheeks",),
        text_hash="text",
        content_hash="card-large-lips-soft-cheeks",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:weak, closer_to_liked_cluster. face_detail: lips: large, "
            "brows: neat/normal, cheeks: soft, lower_third: soft/thin. "
            "Natural/cute, но нет slim cheeks + thin lower_third."
        )
    )
    observation = MediaObservation(
        card_hash="card-large-lips-soft-cheeks",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "доминирующие крупные губы" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "доминирующие крупные губы" in decision.reasons


def test_autolike_decision_allows_large_lips_with_neat_brows_and_non_round_face() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("liked-large-lips-neat-brows",),
        text_hash="text",
        content_hash="card-liked-large-lips-neat-brows",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "лицо видно, face_match:weak, closer_to_liked_cluster, "
            "quality_limited_face_match. face_detail: lips: large, "
            "lip_expression: natural, brows: neat/thin, cheeks: slim, "
            "lower_third: thin, face_width: narrow. Губы заметные и крупные, "
            "но брови аккуратные/тонкие, лицо не округлое и нижняя треть тонкая, "
            "не массивная. Компактная liked-геометрия лица: компактная "
            "центральная зона, большие открытые округлые глаза, короткий "
            "аккуратный нос."
        )
    )
    observation = MediaObservation(
        card_hash="card-liked-large-lips-neat-brows",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "тяжелая связка губ-бровей-щек" not in tags
    assert "массивная нижняя треть и огромные губы" not in tags
    assert "доминирующие крупные губы" not in tags
    assert "округло-пухловатое лицо" not in tags
    assert "face_match:mismatch" not in tags
    assert "компактная liked-геометрия лица" in tags
    assert decision.action == "like"


def test_vision_tags_do_not_soften_explicit_rough_face_to_weak() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:mismatch. Кадр немного размытый, но лицо не понравилось: "
        "жесткие черты лица, тяжелая зона бровей и грубоватое лицо."
    )

    assert "face_match:mismatch" in tags
    assert "face_match:weak" not in tags
    assert "грубое лицо" in tags


def test_vision_tags_disliked_cluster_does_not_create_liked_cluster_tag() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:mismatch, closer_to_disliked_cluster, disliked_cluster_face, "
        "недостаточно doll-like лицо, грубое лицо"
    )

    assert "disliked_cluster_face" in tags
    assert "liked_cluster_face" not in tags


def test_vision_tags_parse_reference_class_as_disliked_cluster() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "reference_classification:\n"
            "face_reference_class: disliked_face\n"
            "nearest_reference_side: disliked_face_2\n"
            "reference_confidence: 0.72\n"
            "face_match:weak, natural/cute, но rejected natural-cute cluster."
        )
    )
    card = TasteCard(
        message_ids=("reference-class-disliked",),
        text_hash="text",
        content_hash="card-reference-class-disliked",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-reference-class-disliked",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "disliked_cluster_face" in tags
    assert "face_match:mismatch" in tags
    assert decision.action == "skip"
    assert "disliked_cluster_face" in decision.reasons


def test_vision_tags_parse_reference_class_as_disliked_body_stop() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    tags = tuple(
        _vision_tags(
            "reference_classification:\n"
            "face_reference_class: unknown\n"
            "body_reference_class: disliked_body\n"
            "nearest_reference_side: disliked_body_1\n"
            "face_match:unknown, body-first/mirror-first."
        )
    )
    card = TasteCard(
        message_ids=("reference-class-disliked-body",),
        text_hash="text",
        content_hash="card-reference-class-disliked-body",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-reference-class-disliked-body",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "disliked_body_reference" in tags
    assert decision.action == "skip"
    assert "disliked_body_reference" in decision.reasons


def test_vision_tags_transport_does_not_trigger_sport_style() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _vision_tags

    tags = _vision_tags(
        "face_match:weak, closer_to_liked_cluster. Фон похож на салон транспорта, "
        "hoodie/casual, natural."
    )

    assert "спортивный стиль" not in tags


def test_autolike_decision_skips_disliked_cluster_face_even_if_marked_strong() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("disliked-cluster-strong",),
        text_hash="text",
        content_hash="card-disliked-cluster-strong",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:strong, но calibrated_cluster: closer_to_disliked_cluster. "
            "Лицо natural/cute, но по калибровке ближе к rejected natural-cute "
            "кластеру, чем к liked-кластеру."
        )
    )
    observation = MediaObservation(
        card_hash="card-disliked-cluster-strong",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "disliked_cluster_face" in tags
    assert decision.action == "skip"
    assert "disliked_cluster_face" in decision.reasons


def test_autolike_decision_skips_partly_hidden_full_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("yuki-full-face",),
        text_hash="text",
        content_hash="card-yuki-full-face",
        age=19,
        city="Москва",
        text_terms=("ночная жизнь",),
    )
    tags = tuple(
        _vision_tags(
            "лицо частично закрыто предметом, но видимая нижняя часть лица "
            "и щеки выглядят полными/округлыми; face_match:weak, natural"
        )
    )
    observation = MediaObservation(
        card_hash="card-yuki-full-face",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "полное лицо" in tags
    assert decision.action == "skip"
    assert "hard_reject_visual_stop" in decision.reasons
    assert "полное лицо" in decision.reasons


def test_autolike_decision_likes_romantic_polymath_with_classic_body_first_style() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("203",),
        text_hash="text",
        content_hash="card-caine",
        age=18,
        city="Москва",
        text_terms=(
            "творчество/искусство",
            "разносторонность/развитие",
            "романтичность/родственная душа",
            "забота/поддержка",
            "языки",
            "музыка",
            "кофе/еда",
        ),
    )
    observation = MediaObservation(
        card_hash="card-caine",
        media_hash="m1",
        kind="photo",
        tags=(
            "body-first/mirror-first",
            "лицо закрыто",
            "классический/formal стиль",
            "dark academia стиль",
            "эстетичный outfit",
            "естественная женственная фигура",
            "большая грудь как главный акцент",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert decision.score >= 8
    assert "body_first_allowed_by_visual_fit" in decision.reasons
    assert "chest_emphasis_allowed:natural_non_vulgar" in decision.reasons
    assert "positive_text:романтичность/родственная душа" in decision.reasons


def test_autolike_decision_likes_cute_fragile_natural_chest_emphasis() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("204",),
        text_hash="text",
        content_hash="card-varya",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-varya",
        media_hash="m1",
        kind="photo",
        tags=(
            "body-first/mirror-first",
            "лицо закрыто",
            "естественный/cute вайб",
            "стройная/хрупкая фигура",
            "аккуратный женственный акцент",
            "уютный casual стиль",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert "body_first_allowed_by_visual_fit" in decision.reasons
    assert "positive_visual:аккуратный женственный акцент" in decision.reasons


def test_autolike_decision_keeps_no_face_body_vibe_exception_like() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("2041",),
        text_hash="text",
        content_hash="card-no-face-body-vibe",
        age=18,
        city="Москва",
    )
    observation = MediaObservation(
        card_hash="card-no-face-body-vibe",
        media_hash="m1",
        kind="photo",
        tags=(
            "body-first/mirror-first",
            "лицо закрыто",
            "стройная/хрупкая фигура",
            "аккуратный женственный акцент",
            "уютный casual стиль",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert "body_first_allowed_by_visual_fit" in decision.reasons


def test_autolike_decision_likes_body_fit_when_face_unknown_or_closed() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("body-fit-face-unknown",),
        text_hash="text",
        content_hash="card-body-fit-face-unknown",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:unknown, лицо закрыто телефоном/ракурсом, "
            "body-first/mirror-first, стройная/хрупкая фигура, "
            "естественная женственная фигура, аккуратный женственный акцент, "
            "уютный casual стиль."
        )
    )
    observation = MediaObservation(
        card_hash="card-body-fit-face-unknown",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "face_match:unknown" in tags
    assert "face_match:weak" not in tags
    assert decision.action == "like"
    assert "body_first_allowed_by_visual_fit" in decision.reasons


def test_autolike_decision_keeps_no_face_body_card_like_when_vision_mentions_manual_cluster() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("body-fit-face-cluster-manual",),
        text_hash="text",
        content_hash="card-body-fit-face-cluster-manual",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "face_match:unknown, лицо обрезано, лицо нечитаемо, "
            "closer_to_disliked_cluster: manual, недостаточно данных по лицу. "
            "body-first/mirror-first, natural/cute, эстетичный outfit, "
            "стройная/хрупкая фигура, уютный casual стиль, серые спортивные штаны."
        )
    )
    observation = MediaObservation(
        card_hash="card-body-fit-face-cluster-manual",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "face_match:unknown" in tags
    assert "face_match:mismatch" not in tags
    assert "disliked_cluster_face" not in tags
    assert "спортивный стиль" not in tags
    assert decision.action == "like"


def test_autolike_decision_does_not_let_strong_text_bypass_visual_gate() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("205",),
        text_hash="text",
        content_hash="card-text-only-strong",
        age=18,
        city="Москва",
        text_terms=(
            "творчество/искусство",
            "разносторонность/развитие",
            "романтичность/родственная душа",
            "забота/поддержка",
        ),
    )
    observation = MediaObservation(
        card_hash="card-text-only-strong",
        media_hash="m1",
        kind="photo",
        tags=("body-first/mirror-first", "лицо закрыто"),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "manual"
    assert "visual_gate_not_passed" in decision.reasons


def test_autolike_decision_skips_non_human_meme_media_without_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("unicorn-meme",),
        text_hash="text",
        content_hash="card-unicorn-meme",
        age=18,
        city="Москва",
    )
    tags = tuple(
        _vision_tags(
            "это мем-картинка: игрушечный единорог/лошадь, не фото человека, "
            "лица человека нет, face_match:unknown, quirky"
        )
    )
    observation = MediaObservation(
        card_hash="card-unicorn-meme",
        media_hash="m1",
        kind="photo",
        tags=tags,
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "не фото человека" in tags
    assert decision.action == "skip"
    assert decision.reasons == ("non_human_or_irrelevant_media",)


def test_autolike_decision_skips_landscape_and_dog_media_without_human_face() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        _vision_tags,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("street-and-dog",),
        text_hash="text",
        content_hash="card-street-and-dog",
        age=18,
        city="Москва",
    )
    observations = (
        MediaObservation(
            card_hash="card-street-and-dog",
            media_hash="m1",
            kind="photo",
            tags=tuple(
                _vision_tags(
                    "темная улица, дорога без человека, пейзаж, no person, "
                    "face_match:unknown"
                )
            ),
        ),
        MediaObservation(
            card_hash="card-street-and-dog",
            media_hash="m2",
            kind="photo",
            tags=tuple(
                _vision_tags(
                    "собака в транспорте, животное, лица человека нет, "
                    "face_match:unknown"
                )
            ),
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=observations)

    assert all("не фото человека" in observation.tags for observation in observations)
    assert decision.action == "skip"
    assert decision.reasons == ("non_human_or_irrelevant_media",)


def test_autolike_decision_skips_antifreeze_self_harm_joke_text() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        decide_daivinchik_autolike,
        normalize_history_to_cards,
    )

    text = "Аня, 18, Москва – Пью антифриз с тоской на детские пособия"
    card = normalize_history_to_cards(
        {"messages": [{"id": 1, "text": text, "media": [{"id": "photo-1"}]}]}
    )[0]
    observation = MediaObservation(
        card_hash=card.content_hash,
        media_hash="m1",
        kind="photo",
        tags=("face_match:weak", "естественный/cute вайб", "лицо видно"),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "опасный/self-harm юмор" in card.text_terms
    assert decision.action == "skip"
    assert decision.reasons == ("hard_text_stop", "опасный/self-harm юмор")


def test_autolike_decision_skips_profiles_not_seeking_relationships() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        decide_daivinchik_autolike,
        normalize_history_to_cards,
    )

    texts = (
        "Anastasia, 19, Москва – если ты ищешь отношений, то можешь не лайкать)",
        ("Никс, 18, Москва – Отношений не ищу. Ищу просто общение, интересных людей."),
        (
            "яся, 18, Москва – тут только чтобы найти друзей, ничего больше. "
            "если хотите лайкать по поводу отношений, катитесь"
        ),
    )
    for text in texts:
        card = normalize_history_to_cards(
            {"messages": [{"id": 1, "text": text, "media": [{"id": "photo-1"}]}]}
        )[0]
        observation = MediaObservation(
            card_hash=card.content_hash,
            media_hash="m1",
            kind="photo",
            tags=(
                "face_match:strong",
                "liked_cluster_face",
                "естественный/cute вайб",
                "стройная/хрупкая фигура",
            ),
        )

        decision = decide_daivinchik_autolike(card, observations=(observation,))

        assert "не ищет отношений" in card.text_terms
        assert decision.action == "skip"
        assert decision.reasons == ("hard_text_stop", "не ищет отношений")


def test_autolike_decision_skips_declared_underage_even_if_profile_age_is_18() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        decide_daivinchik_autolike,
        normalize_history_to_cards,
    )

    text = (
        "яся, 18, Москва – МНЕ 13!! стоит 18 потому что дв не пропускает мя по другому"
    )
    card = normalize_history_to_cards(
        {"messages": [{"id": 1, "text": text, "media": [{"id": "photo-1"}]}]}
    )[0]
    observation = MediaObservation(
        card_hash=card.content_hash,
        media_hash="m1",
        kind="photo",
        tags=(
            "face_match:strong",
            "liked_cluster_face",
            "естественный/cute вайб",
            "стройная/хрупкая фигура",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert card.age == 18
    assert "самоописание младше 18" in card.text_terms
    assert decision.action == "skip"
    assert decision.reasons == ("hard_text_stop", "самоописание младше 18")


def test_autolike_decision_skips_med_only_profiles() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        decide_daivinchik_autolike,
        normalize_history_to_cards,
    )

    text = "Софья, 18, Москва – может\nесть кто из меда?"
    card = normalize_history_to_cards(
        {"messages": [{"id": 1, "text": text, "media": [{"id": "photo-1"}]}]}
    )[0]
    observation = MediaObservation(
        card_hash=card.content_hash,
        media_hash="m1",
        kind="photo",
        tags=(
            "face_match:strong",
            "liked_cluster_face",
            "естественный/cute вайб",
            "стройная/хрупкая фигура",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert "только мед/профессия" in card.text_terms
    assert decision.action == "skip"
    assert decision.reasons == ("hard_text_stop", "только мед/профессия")


def test_autolike_decision_ignores_smoking_text() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        MediaObservation,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("206",),
        text_hash="text",
        content_hash="card-maria",
        age=19,
        city="Москва",
        text_terms=(
            "творчество/искусство",
            "разносторонность/развитие",
            "музыка",
            "игры/онлайн",
        ),
    )
    observation = MediaObservation(
        card_hash="card-maria",
        media_hash="m1",
        kind="photo",
        tags=(
            "лицо видно",
            "face_match:strong",
            "естественный/cute вайб",
            "стройная/хрупкая фигура",
            "эстетичный outfit",
            "body-first/mirror-first",
        ),
    )

    decision = decide_daivinchik_autolike(card, observations=(observation,))

    assert decision.action == "like"
    assert not any("курение" in reason for reason in decision.reasons)
    assert "body_first_allowed_by_visual_fit" in decision.reasons


def test_daivinchik_text_terms_capture_romance_and_development_without_status_penalty() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import _text_terms

    text = (
        "я — невероятно разносторонний человек и сочетаю в себе самые "
        "разнообразные увлечения. обожаю классический стиль в одежде, театры, "
        "кинематограф, музеи; играю на синтезаторе, выкладываю свои мысли на "
        "бумаге в виде произведений, стихов и наполненных смыслом рисунков; "
        "готовлю ресторанные блюда и пару месяцев назад прошла обучающий курс "
        "на бармена, поэтому хорошо разбираюсь в алкоголе; ценю хорошую "
        "концептуальную музыку; изучаю китайский, японский и английский языки. "
        "помимо всего вышеперечисленного, я мечтаю встретить любовь всей своей "
        "жизни, свою родственную душу. желаю разделить каждый радостный или, "
        "наоборот, тоскливый момент нашей с этим человеком жизни; горю ярким "
        "пламенем по отношению к подаркам, заботе и поддержке."
    )

    terms = set(_text_terms(text))

    assert "творчество/искусство" in terms
    assert "разносторонность/развитие" in terms
    assert "романтичность/родственная душа" in terms
    assert "забота/поддержка" in terms
    assert "языки" in terms
    assert "музыка" in terms
    assert "ночная жизнь" not in terms
    assert "статусность/меркантильность" not in terms


def test_daivinchik_text_terms_ignore_smoking() -> None:
    from src.agent_runtime.workers.daivinchik_profile import _text_terms

    terms = set(_text_terms("P.s. кому принципиально, то я курю"))

    assert "курение" not in terms


def test_autolike_decision_stops_on_attention_case_before_scoring() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        AttentionCase,
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("104",),
        text_hash="text",
        content_hash="card-attention",
        age=18,
    )
    attention = AttentionCase(
        message_id="104",
        kind="identity_verification",
        text_hash="hash",
        excerpt="Подтвердите личность видео",
    )

    decision = decide_daivinchik_autolike(card, attention_cases=(attention,))

    assert decision.action == "attention_required"
    assert decision.confidence == 1.0
    assert "attention_required:identity_verification" in decision.reasons


def test_autolike_decision_routes_missing_visual_signal_to_manual_review() -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        TasteCard,
        decide_daivinchik_autolike,
    )

    card = TasteCard(
        message_ids=("105",),
        text_hash="text",
        content_hash="card-no-vision",
        age=18,
        text_terms=("кофе/еда",),
    )

    decision = decide_daivinchik_autolike(card)

    assert decision.action == "manual"
    assert decision.score == 2
    assert decision.confidence == 0.8
    assert "missing_visual_signal" in decision.reasons
    assert "positive_text:кофе/еда" in decision.reasons
    assert "preferred_age:18-20" in decision.reasons


@pytest.mark.asyncio
async def test_terminal_codex_vision_uses_codex_exec_image_and_no_api_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        TerminalCodexVisionDescriber,
    )

    image_path = tmp_path / "profile.jpg"
    image_path.write_bytes(b"fake-image")
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    async def fake_process_factory(
        *args: str, **kwargs: Any
    ) -> _FakeCodexVisionProcess:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        output_path = Path(args[args.index("--output-last-message") + 1])
        image_arg = Path(args[args.index("--image") + 1])
        assert image_arg.exists()
        assert image_arg.parent == output_path.parent
        process = _FakeCodexVisionProcess(
            returncode=0,
            output_path=output_path,
            final="лицо видно, естественный cute вайб",
        )
        captured["process"] = process
        return process

    monkeypatch.setattr(
        "src.agent_runtime.workers.daivinchik_profile._create_process",
        fake_process_factory,
    )

    result = await TerminalCodexVisionDescriber(
        codex_path="codex-test",
        model="gpt-5.5",
        reasoning_effort="medium",
    ).describe_image_file(image_path, prompt="describe", caller="test")

    args = captured["args"]
    assert result == "лицо видно, естественный cute вайб"
    assert args[0] == "codex-test"
    assert args[args.index("--ask-for-approval") + 1] == "never"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--image") + 1] != str(image_path)
    assert args[args.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="medium"' in args
    assert b"describe" in captured["process"].stdin
    assert "OPENAI_API_KEY" not in captured["env"]


@pytest.mark.asyncio
async def test_terminal_codex_vision_attaches_reference_sheets_before_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        TerminalCodexVisionDescriber,
    )

    liked_sheet = tmp_path / "liked.jpg"
    disliked_sheet = tmp_path / "disliked.jpg"
    image_path = tmp_path / "candidate.jpg"
    liked_sheet.write_bytes(b"liked")
    disliked_sheet.write_bytes(b"disliked")
    image_path.write_bytes(b"candidate")
    captured: dict[str, Any] = {}

    async def fake_process_factory(
        *args: str, **kwargs: Any
    ) -> _FakeCodexVisionProcess:
        del kwargs
        captured["args"] = args
        output_path = Path(args[args.index("--output-last-message") + 1])
        image_args = [
            Path(args[index + 1]) for index, arg in enumerate(args) if arg == "--image"
        ]
        captured["image_args"] = image_args
        process = _FakeCodexVisionProcess(
            returncode=0,
            output_path=output_path,
            final="face_match:mismatch, closer_to_disliked_cluster",
        )
        captured["process"] = process
        return process

    monkeypatch.setattr(
        "src.agent_runtime.workers.daivinchik_profile._create_process",
        fake_process_factory,
    )

    result = await TerminalCodexVisionDescriber(
        codex_path="codex-test",
        reference_image_paths=(liked_sheet, disliked_sheet),
    ).describe_image_file(image_path, prompt="describe", caller="test")

    image_args = captured["image_args"]
    assert result == "face_match:mismatch, closer_to_disliked_cluster"
    assert len(image_args) == 3
    assert image_args[0].name.startswith("reference_1")
    assert image_args[1].name.startswith("reference_2")
    assert image_args[2].name.startswith("input")
    assert b"reference sheets" in captured["process"].stdin
    assert b"candidate" in captured["process"].stdin


@pytest.mark.asyncio
async def test_terminal_profile_message_classifier_uses_cheap_codex_exec_no_api_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        TerminalCodexProfileMessageClassifier,
    )

    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    async def fake_process_factory(
        *args: str, **kwargs: Any
    ) -> _FakeProfileClassifierProcess:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        output_path = Path(args[args.index("--output-last-message") + 1])
        process = _FakeProfileClassifierProcess(
            returncode=0,
            output_path=output_path,
            final="profile",
        )
        captured["process"] = process
        return process

    monkeypatch.setattr(
        "src.agent_runtime.workers.daivinchik_profile._create_process",
        fake_process_factory,
    )

    result = await TerminalCodexProfileMessageClassifier(
        codex_path="codex-test",
        model="gpt-5.4-mini",
        reasoning_effort="low",
    ).classify("Аня, 18, Москва – люблю кофе", caller="test")

    args = captured["args"]
    assert result == "profile"
    assert args[0] == "codex-test"
    assert args[args.index("--ask-for-approval") + 1] == "never"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--model") + 1] == "gpt-5.4-mini"
    assert 'model_reasoning_effort="low"' in args
    assert b"\xd0\x90\xd0\xbd\xd1\x8f" in captured["process"].stdin
    assert "OPENAI_API_KEY" not in captured["env"]


@pytest.mark.asyncio
async def test_attention_detection_can_use_terminal_classifier_for_ambiguous_text() -> (
    None
):
    from src.agent_runtime.workers.daivinchik_profile import detect_attention_cases

    class FakeClassifier:
        async def classify(self, text: str, *, caller: str) -> str:
            del caller
            if "анкета" in text:
                return "profile"
            return "non_profile"

    raw_history = {
        "messages": [
            {"id": 1, "text": "анкета: Аня, 18, Москва"},
            {"id": 2, "text": "какой-то системный текст без понятной формы"},
        ]
    }

    cases = await detect_attention_cases(raw_history, classifier=FakeClassifier())

    assert [case.message_id for case in cases] == ["2"]
    assert cases[0].kind == "unknown_non_profile"


@pytest.mark.asyncio
async def test_daivinchik_profile_uses_read_media_only_capabilities() -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_TASTE_PROFILE_READONLY
    from src.agent_runtime.workers.telegram_mcp import build_telegram_mcp_tool_gateway

    class FakeTelegramClient:
        async def list_tools(self) -> tuple[str, ...]:
            return ()

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            return json.dumps({"name": name, "arguments": arguments})

    gateway = build_telegram_mcp_tool_gateway(client=FakeTelegramClient())

    assert DAIVINCHIK_TASTE_PROFILE_READONLY.allows("telegram_mcp_read")
    assert DAIVINCHIK_TASTE_PROFILE_READONLY.allows("telegram_mcp_media_read")
    assert not DAIVINCHIK_TASTE_PROFILE_READONLY.allows("telegram_mcp_send")
    assert not DAIVINCHIK_TASTE_PROFILE_READONLY.allows("telegram_mcp_modify")
    assert not DAIVINCHIK_TASTE_PROFILE_READONLY.allows("telegram_mcp_admin")
    assert not DAIVINCHIK_TASTE_PROFILE_READONLY.allows("telegram_mcp_media_files")

    await gateway.execute(
        DAIVINCHIK_TASTE_PROFILE_READONLY,
        "telegram_mcp_call_read",
        {"tool_name": "get_history", "arguments": {"chat_id": "daivinchik"}},
    )
    await gateway.execute(
        DAIVINCHIK_TASTE_PROFILE_READONLY,
        "telegram_mcp_call_media_read",
        {"tool_name": "get_media_info", "arguments": {"message_id": 1}},
    )

    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            DAIVINCHIK_TASTE_PROFILE_READONLY,
            "telegram_mcp_send_message",
            {"chat_id": "daivinchik", "message": "no"},
        )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            DAIVINCHIK_TASTE_PROFILE_READONLY,
            "telegram_mcp_call_modify",
            {"tool_name": "press_inline_button", "arguments": {"message_id": 1}},
        )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            DAIVINCHIK_TASTE_PROFILE_READONLY,
            "telegram_mcp_call_admin",
            {"tool_name": "ban_user", "arguments": {"user_id": 1}},
        )


@pytest.mark.asyncio
async def test_daivinchik_autolike_profile_exposes_only_dedicated_actions() -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.profiles import (
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
        DAIVINCHIK_AUTOLIKE_MVP,
    )
    from src.agent_runtime.workers.telegram_mcp import build_telegram_mcp_tool_gateway

    class FakeTelegramClient:
        async def list_tools(self) -> tuple[str, ...]:
            return ()

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            return json.dumps({"name": name, "arguments": arguments})

    gateway = build_telegram_mcp_tool_gateway(client=FakeTelegramClient())

    assert DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_read")
    assert DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_media_read")
    assert DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_daivinchik_button")
    assert DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_daivinchik_notify")
    assert DAIVINCHIK_AUTOLIKE_MVP.allows(
        "telegram_mcp_daivinchik_forward_liked_profile"
    )
    assert not DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_modify")
    assert not DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_send")
    assert not DAIVINCHIK_AUTOLIKE_MVP.allows("telegram_mcp_admin")
    assert DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allows("telegram_mcp_daivinchik_button")
    assert DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allows(
        "telegram_mcp_daivinchik_reply_button"
    )
    assert DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allows(
        "telegram_mcp_daivinchik_forward_liked_profile"
    )
    assert DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allows("telegram_mcp_daivinchik_notify")
    assert not DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allows("telegram_mcp_send")

    approval = AgentToolApproval.approved(
        approval_id="daivinchik-test",
        capabilities=(
            "telegram_mcp_daivinchik_button",
            "telegram_mcp_daivinchik_notify",
            "telegram_mcp_daivinchik_forward_liked_profile",
        ),
        approved_by=12345,
    )
    await gateway.execute(
        DAIVINCHIK_AUTOLIKE_MVP,
        "telegram_mcp_daivinchik_press_inline_button",
        {"chat_id": "daivinchik", "message_id": 1, "button_text": "❤️"},
        approval=approval,
    )
    await gateway.execute(
        DAIVINCHIK_AUTOLIKE_MVP,
        "telegram_mcp_daivinchik_notify",
        {"chat_id": "@KoTTiH", "message": "stop"},
        approval=approval,
    )
    await gateway.execute(
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
        "telegram_mcp_daivinchik_reply_button",
        {"chat_id": "daivinchik", "button_text": "1 🚀"},
        approval=AgentToolApproval.approved(
            approval_id="daivinchik-test",
            capabilities=("telegram_mcp_daivinchik_reply_button",),
            approved_by=12345,
        ),
    )
    await gateway.execute(
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
        "telegram_mcp_daivinchik_forward_liked_profile",
        {
            "from_chat_id": "daivinchik",
            "to_chat_id": "@KoTTiH",
            "message_ids": ("1",),
        },
        approval=AgentToolApproval.approved(
            approval_id="daivinchik-test",
            capabilities=("telegram_mcp_daivinchik_forward_liked_profile",),
            approved_by=12345,
        ),
    )
    await gateway.execute(
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
        "telegram_mcp_daivinchik_notify",
        {"chat_id": "@KoTTiH", "message": "manual stop"},
        approval=AgentToolApproval.approved(
            approval_id="daivinchik-test",
            capabilities=("telegram_mcp_daivinchik_notify",),
            approved_by=12345,
        ),
    )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            DAIVINCHIK_AUTOLIKE_MVP,
            "telegram_mcp_call_modify",
            {"tool_name": "delete_message", "arguments": {"message_id": 1}},
        )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            DAIVINCHIK_AUTOLIKE_MVP,
            "telegram_mcp_send_message",
            {"chat_id": "@KoTTiH", "message": "broad send denied"},
        )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
            "telegram_mcp_daivinchik_reply_button",
            {"chat_id": "daivinchik", "button_text": "подтвердить"},
            approval=AgentToolApproval.approved(
                approval_id="daivinchik-test",
                capabilities=("telegram_mcp_daivinchik_reply_button",),
                approved_by=12345,
            ),
        )


@pytest.mark.asyncio
async def test_media_lifecycle_photo_video_frame_and_cleanup(tmp_path: Path) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(payload: dict[str, Any]) -> str:
        assert payload["tool_name"] == "get_history"
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 10,
                        "text": "Маша, 24\nМосква\nсырой приватный текст анкеты",
                        "media": [{"id": "photo-10", "type": "photo"}],
                    },
                    {"id": 11, "media": [{"id": "video-11", "type": "video"}]},
                    {"id": 12, "text": "лайк"},
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        tool_name = payload["tool_name"]
        message_id = int(payload["arguments"]["message_id"])
        if tool_name == "get_media_info":
            media_type = "video" if message_id == 11 else "photo"
            return json.dumps({"media_type": media_type})
        file_path = Path(payload["arguments"]["file_path"])
        suffix = ".mp4" if message_id == 11 else ".jpg"
        target = file_path.with_suffix(suffix)
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
        )
    )
    vision = FakeVision()
    frame_extractor = FakeFrameExtractor()
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=vision,
        frame_extractor=frame_extractor,
    )
    pack = ContextPack(user_request=json.dumps({"chat_id": "daivinchik-chat"}))

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    report_path = tmp_path / "social" / "daivinchik" / "taste_profile.md"
    temp_root = tmp_path / "telegram-mcp" / "daivinchik-profile"
    assert capsule.summary == "Daivinchik taste profile completed."
    assert report_path.exists()
    assert not any(temp_root.iterdir())
    assert len(vision.calls) == 2
    assert len(frame_extractor.calls) == 1
    assert "сырой приватный текст анкеты" not in report_path.read_text(encoding="utf-8")
    assert "Карточек с фото: 1" in capsule.markdown_report
    assert "Карточек с видео: 1" in capsule.markdown_report


@pytest.mark.asyncio
async def test_terminal_vision_worker_path_processes_media_without_llm_api(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 30,
                        "text": "Аня, 18, Москва – короткий милый текст",
                        "media": [{"id": "photo-30", "type": "photo"}],
                    },
                    {"id": 31, "text": "❤️"},
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
        )
    )
    terminal_vision = FakeTerminalVision()
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=terminal_vision,
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(user_request=json.dumps({"chat_id": "daivinchik-chat"}))

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    assert capsule.summary == "Daivinchik taste profile completed."
    assert len(terminal_vision.calls) == 1
    report = (tmp_path / "social" / "daivinchik" / "taste_profile.md").read_text(
        encoding="utf-8"
    )
    assert "В лайкнутых media повторяется" in report


@pytest.mark.asyncio
async def test_autolike_decision_mode_scores_only_current_card(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 50,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-50", "type": "photo"}],
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_decision"}
        )
    )

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    assert capsule.summary == "Daivinchik autolike decision: like"
    assert not (tmp_path / "social" / "daivinchik" / "taste_profile.md").exists()
    decision_path = tmp_path / "social" / "daivinchik" / "autolike_decision.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert payload["decision"]["action"] == "like"
    assert payload["card"]["media_count"] == 1
    assert payload["observations"][0]["tags"]


@pytest.mark.asyncio
async def test_autolike_live_mode_presses_like_for_current_card(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_MVP
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    pressed: list[dict[str, Any]] = []
    notified: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 60,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 60,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-60", "type": "photo"}],
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def notify(payload: dict[str, Any]) -> str:
        notified.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_notify",
                "telegram_mcp_daivinchik_notify",
                notify,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {
            "chat_id": "daivinchik-chat",
            "mode": "autolike_live",
            "max_actions": 1,
            "notify_chat_id": "@KoTTiH",
        }
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_MVP),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert pressed == [
        {"chat_id": "daivinchik-chat", "message_id": "60", "button_text": "❤️"}
    ]
    assert notified == []
    audit_path = tmp_path / "social" / "daivinchik" / "autolike_live.jsonl"
    audit = audit_path.read_text(encoding="utf-8")
    assert '"decision":"like"' in audit


@pytest.mark.asyncio
async def test_autolike_live_mode_uses_reply_button_for_non_callback_decision(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_MVP
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    pressed: list[dict[str, Any]] = []
    reply_buttons: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 60,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": False},
                        {"index": 1, "text": "❤️", "has_callback": False},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 60,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-60", "type": "photo"}],
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def reply_button(payload: dict[str, Any]) -> str:
        reply_buttons.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {"chat_id": "daivinchik-chat", "mode": "autolike_live", "max_actions": 1}
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_MVP),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert pressed == []
    assert reply_buttons == [{"chat_id": "daivinchik-chat", "button_text": "❤️"}]
    audit_path = tmp_path / "social" / "daivinchik" / "autolike_live.jsonl"
    audit = audit_path.read_text(encoding="utf-8")
    assert '"decision":"like"' in audit


@pytest.mark.asyncio
async def test_autolike_live_mode_starts_from_waiting_menu_with_reply_button(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    history_reads = 0
    reply_buttons: list[dict[str, Any]] = []
    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        nonlocal history_reads
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 61,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        history_reads += 1
        if history_reads == 1:
            return json.dumps(
                {
                    "messages": [
                        {
                            "id": 50,
                            "text": (
                                "Подождем пока кто-то увидит твою анкету\n\n"
                                "1. Смотреть анкеты.\n2. Моя анкета.\n"
                                "3. Я больше не хочу никого искать.\n***\n"
                                "4. Активируй Premium — будь в топе ⭐"
                            ),
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 50,
                        "text": (
                            "Подождем пока кто-то увидит твою анкету\n\n"
                            "1. Смотреть анкеты.\n2. Моя анкета.\n"
                            "3. Я больше не хочу никого искать.\n***\n"
                            "4. Активируй Premium — будь в топе ⭐"
                        ),
                    },
                    {"id": 51, "text": "1 🚀"},
                    {"id": 52, "text": "✨🔍"},
                    {
                        "id": 61,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-61", "type": "photo"}],
                    },
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def reply_button(payload: dict[str, Any]) -> str:
        reply_buttons.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_live", "max_actions": 1}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command",
            "agent_tool_approval_capabilities": (
                "telegram_mcp_daivinchik_button,telegram_mcp_daivinchik_reply_button"
            ),
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert reply_buttons == [{"chat_id": "daivinchik-chat", "button_text": "1 🚀"}]
    assert pressed == [
        {"chat_id": "daivinchik-chat", "message_id": "61", "button_text": "❤️"}
    ]
    audit = (tmp_path / "social" / "daivinchik" / "autolike_live.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"decision":"service_menu_start"' in audit
    assert '"decision":"like"' in audit


@pytest.mark.asyncio
async def test_autolike_live_mode_starts_from_profile_edit_menu_with_premium(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    history_reads = 0
    reply_buttons: list[dict[str, Any]] = []
    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        nonlocal history_reads
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 61,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        history_reads += 1
        if history_reads == 1:
            return json.dumps(
                {
                    "messages": [
                        {
                            "id": 50,
                            "text": (
                                "1. Смотреть анкеты.\n"
                                "2. Заполнить анкету заново.\n"
                                "3. Изменить фото/видео.\n"
                                "4. Изменить текст анкеты.\n"
                                "***\n"
                                "5. Активируй Premium — будь в топе ⭐️."
                            ),
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 50,
                        "text": (
                            "1. Смотреть анкеты.\n"
                            "2. Заполнить анкету заново.\n"
                            "3. Изменить фото/видео.\n"
                            "4. Изменить текст анкеты.\n"
                            "***\n"
                            "5. Активируй Premium — будь в топе ⭐️."
                        ),
                    },
                    {"id": 51, "text": "1 🚀"},
                    {"id": 52, "text": "✨🔍"},
                    {
                        "id": 61,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-61", "type": "photo"}],
                    },
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def reply_button(payload: dict[str, Any]) -> str:
        reply_buttons.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_live", "max_actions": 1}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command",
            "agent_tool_approval_capabilities": (
                "telegram_mcp_daivinchik_button,telegram_mcp_daivinchik_reply_button"
            ),
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert reply_buttons == [{"chat_id": "daivinchik-chat", "button_text": "1 🚀"}]
    assert pressed == [
        {"chat_id": "daivinchik-chat", "message_id": "61", "button_text": "❤️"}
    ]
    audit = (tmp_path / "social" / "daivinchik" / "autolike_live.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"decision":"service_menu_start"' in audit
    assert '"decision":"attention_required"' not in audit


@pytest.mark.asyncio
async def test_autolike_live_mode_does_not_start_menu_when_profile_is_visible(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    reply_buttons: list[dict[str, Any]] = []
    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 61,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 50,
                        "text": (
                            "Подождем пока кто-то увидит твою анкету\n\n"
                            "1. Смотреть анкеты.\n2. Моя анкета.\n"
                            "3. Я больше не хочу никого искать.\n***\n"
                            "4. Активируй Premium — будь в топе ⭐"
                        ),
                    },
                    {"id": 51, "text": "1 🚀"},
                    {"id": 52, "text": "✨🔍"},
                    {
                        "id": 61,
                        "text": "даша, 19, москва",
                        "media": [{"id": "photo-61", "type": "photo"}],
                    },
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def reply_button(payload: dict[str, Any]) -> str:
        reply_buttons.append(payload)
        return "must not be used"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_live", "max_actions": 1}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command",
            "agent_tool_approval_capabilities": (
                "telegram_mcp_daivinchik_button,telegram_mcp_daivinchik_reply_button"
            ),
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert reply_buttons == []
    assert pressed == [
        {"chat_id": "daivinchik-chat", "message_id": "61", "button_text": "❤️"}
    ]


@pytest.mark.asyncio
async def test_autolike_live_mode_stops_on_missing_visual_signal_without_dislike(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 2141,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 2141,
                        "text": "даша, 19, москва",
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_live", "max_actions": 1}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command",
            "agent_tool_approval_capabilities": (
                "telegram_mcp_daivinchik_button,telegram_mcp_daivinchik_reply_button"
            ),
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live stopped."
    assert pressed == []
    audit = (tmp_path / "social" / "daivinchik" / "autolike_live.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"message_id":"2141"' in audit
    assert '"decision":"manual"' in audit
    assert "missing_visual_signal" in audit
    assert "preferred_age:18-20" in audit
    assert "preferred_city:moscow" in audit


@pytest.mark.asyncio
async def test_autolike_live_mode_recovers_media_from_nearby_message_id(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    media_info_message_ids: list[str] = []
    downloaded_message_ids: list[str] = []
    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 2147,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 2147,
                        "text": "Аня, 19, Москва",
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        message_id = str(payload["arguments"]["message_id"])
        if payload["tool_name"] == "get_media_info":
            media_info_message_ids.append(message_id)
            if message_id == "2146":
                return "MessageMediaPhoto(photo=Photo(...), video_sizes=[])"
            return "No media found in the specified message."
        downloaded_message_ids.append(message_id)
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_live", "max_actions": 1}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command",
            "agent_tool_approval_capabilities": (
                "telegram_mcp_daivinchik_button,telegram_mcp_daivinchik_reply_button"
            ),
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert media_info_message_ids == ["2147", "2146", "2145", "2146"]
    assert downloaded_message_ids == ["2146"]
    assert pressed == [
        {"chat_id": "daivinchik-chat", "message_id": "2147", "button_text": "❤️"}
    ]
    audit = (tmp_path / "social" / "daivinchik" / "autolike_live.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"decision":"like"' in audit
    assert "media_retry_found_nearby" in audit
    assert "media_retry_vision_ok" in audit
    assert "missing_visual_signal" not in audit


@pytest.mark.asyncio
async def test_autolike_live_mode_notifies_and_stops_on_attention(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_MVP
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    notified: list[dict[str, Any]] = []

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 70,
                        "text": "Подтверди личность: срочно отправь видео с лицом",
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def press(_payload: dict[str, Any]) -> str:
        raise AssertionError("must not press buttons after attention case")

    async def notify(payload: dict[str, Any]) -> str:
        notified.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_notify",
                "telegram_mcp_daivinchik_notify",
                notify,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {
            "chat_id": "daivinchik-chat",
            "mode": "autolike_live",
            "max_actions": 3,
            "notify_chat_id": "@KoTTiH",
        }
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_MVP),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik attention required."
    assert notified
    assert notified[0]["chat_id"] == "@KoTTiH"
    assert "identity_verification" in notified[0]["message"]


@pytest.mark.asyncio
async def test_autolike_live_stops_on_verification_voice_or_video_message_words(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_MVP
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    notified: list[dict[str, Any]] = []

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 71,
                        "text": (
                            "Верификация Дайвинчик: отвечать нужно кружочком, "
                            "видеосообщение или голосовое сообщение."
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def press(_payload: dict[str, Any]) -> str:
        raise AssertionError("must not press buttons on verification wording")

    async def notify(payload: dict[str, Any]) -> str:
        notified.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_notify",
                "telegram_mcp_daivinchik_notify",
                notify,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {
            "chat_id": "daivinchik-chat",
            "mode": "autolike_live",
            "max_actions": 3,
            "notify_chat_id": "@KoTTiH",
        }
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_MVP),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik attention required."
    assert notified
    assert "identity_verification" in notified[0]["message"]


@pytest.mark.asyncio
async def test_autolike_stop_mode_presses_sleep_inline_button(tmp_path: Path) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": 72,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                        {"index": 2, "text": "💤", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 72,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-72", "type": "photo"}],
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def reply_button(_payload: dict[str, Any]) -> str:
        raise AssertionError("stop mode must not send reply-keyboard text")

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_stop"}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command-stop",
            "agent_tool_approval_capabilities": "telegram_mcp_daivinchik_button",
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live stopped."
    assert pressed == [
        {"chat_id": "daivinchik-chat", "message_id": "72", "button_text": "💤"}
    ]


@pytest.mark.asyncio
async def test_autolike_stop_mode_does_not_press_anything_on_verification(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import ContextPack
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 73,
                        "text": (
                            "Верификация Дайвинчик: отвечать нужно кружочком, "
                            "видеосообщение или голосовое сообщение."
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def press(_payload: dict[str, Any]) -> str:
        raise AssertionError("must not press buttons after verification")

    async def reply_button(_payload: dict[str, Any]) -> str:
        raise AssertionError("must not send reply-keyboard text after verification")

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "mode": "autolike_stop"}
        ),
        metadata={
            "agent_tool_approval_id": "daivinchik-bot-command-stop",
            "agent_tool_approval_capabilities": "telegram_mcp_daivinchik_button",
        },
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik attention required."


@pytest.mark.asyncio
async def test_autolike_live_mode_repeats_until_max_actions(tmp_path: Path) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_MVP
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    history_reads = 0
    pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        nonlocal history_reads
        if payload["tool_name"] == "list_inline_buttons":
            message_id = str(payload["arguments"]["message_id"])
            return json.dumps(
                {
                    "message_id": message_id,
                    "results": [
                        {"index": 0, "text": "👎", "has_callback": True},
                        {"index": 1, "text": "❤️", "has_callback": True},
                    ],
                },
                ensure_ascii=False,
            )
        history_reads += 1
        message_id = 80 + history_reads
        return json.dumps(
            {
                "messages": [
                    {
                        "id": message_id,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": f"photo-{message_id}", "type": "photo"}],
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(payload: dict[str, Any]) -> str:
        pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    async def notify(_payload: dict[str, Any]) -> str:
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_notify",
                "telegram_mcp_daivinchik_notify",
                notify,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {
            "chat_id": "daivinchik-chat",
            "mode": "autolike_live",
            "max_actions": 2,
            "notify_chat_id": "@KoTTiH",
        }
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_MVP),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert [item["message_id"] for item in pressed] == ["81", "82"]


@pytest.mark.asyncio
async def test_autolike_live_restarts_when_newer_waiting_menu_follows_stale_card(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    history_reads = 0
    reply_pressed: list[dict[str, Any]] = []
    inline_pressed: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        nonlocal history_reads
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": str(payload["arguments"]["message_id"]),
                    "results": [{"index": 0, "text": "❤️", "has_callback": True}],
                },
                ensure_ascii=False,
            )

        history_reads += 1
        if history_reads == 1:
            messages = [
                {"id": 10, "text": "Подождем пока кто-то увидит твою анкету"},
                {
                    "id": 11,
                    "text": "1. Смотреть анкеты. 2. Моя анкета. 3. Я больше не хочу никого искать.",
                },
            ]
        elif history_reads == 2:
            messages = [
                {
                    "id": 12,
                    "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                    "media": [{"id": "photo-12", "type": "photo"}],
                }
            ]
        elif history_reads == 3:
            messages = [
                {
                    "id": 12,
                    "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                    "media": [{"id": "photo-12", "type": "photo"}],
                },
                {"id": 13, "text": "Подождем пока кто-то увидит твою анкету"},
                {
                    "id": 14,
                    "text": "1. Смотреть анкеты. 2. Моя анкета. 3. Я больше не хочу никого искать.",
                },
            ]
        else:
            messages = [
                {
                    "id": 15,
                    "text": "Даша, 19, Москва – кофе дома и короткий милый текст",
                    "media": [{"id": "photo-15", "type": "photo"}],
                }
            ]
        return json.dumps({"messages": messages}, ensure_ascii=False)

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def reply_button(payload: dict[str, Any]) -> str:
        reply_pressed.append(payload)
        return json.dumps({"result": "Message sent successfully."})

    async def inline_button(payload: dict[str, Any]) -> str:
        inline_pressed.append(payload)
        return json.dumps({"response": "Button pressed successfully."})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                reply_button,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                inline_button,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {
            "chat_id": "daivinchik-chat",
            "mode": "autolike_live",
            "max_actions": 2,
        }
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert [item["button_text"] for item in reply_pressed] == ["1 🚀", "1 🚀"]
    assert [item["message_id"] for item in inline_pressed] == ["12", "15"]


@pytest.mark.asyncio
async def test_autolike_live_forwards_liked_profile_to_owner_account(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.profiles import DAIVINCHIK_AUTOLIKE_BOT_COMMAND
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    forwarded: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "list_inline_buttons":
            return json.dumps(
                {
                    "message_id": str(payload["arguments"]["message_id"]),
                    "results": [{"index": 0, "text": "❤️", "has_callback": True}],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 42,
                        "text": "Аня, 18, Москва – кофе дома и короткий милый текст",
                        "media": [{"id": "photo-42", "type": "photo"}],
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def media(payload: dict[str, Any]) -> str:
        if payload["tool_name"] == "get_media_info":
            return json.dumps({"media_type": "photo"})
        target = Path(payload["arguments"]["file_path"]).with_suffix(".jpg")
        target.write_bytes(b"fake-media")
        return json.dumps({"path": str(target)})

    async def press(_payload: dict[str, Any]) -> str:
        return json.dumps({"response": "Button pressed successfully."})

    async def forward(payload: dict[str, Any]) -> str:
        forwarded.append(payload)
        return json.dumps({"forwarded": payload["message_ids"]})

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                press,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_forward_liked_profile",
                "telegram_mcp_daivinchik_forward_liked_profile",
                forward,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = _approved_pack(
        {
            "chat_id": "@leomatchbot",
            "mode": "autolike_live",
            "max_actions": 1,
            "liked_forward_chat_id": "@KoTTiH",
        }
    )

    capsule = await worker.run(
        job=_job(pack, profile=DAIVINCHIK_AUTOLIKE_BOT_COMMAND),
        context_pack=pack,
    )

    assert capsule.summary == "Daivinchik autolike live completed."
    assert forwarded == [
        {
            "from_chat_id": "@leomatchbot",
            "to_chat_id": "@KoTTiH",
            "message_ids": ("42",),
        }
    ]


@pytest.mark.asyncio
async def test_attention_stop_mode_aborts_before_media_processing(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 40,
                        "text": "Подтверди личность: срочно отправь видео с лицом",
                    },
                    {
                        "id": 41,
                        "text": "Маша, 18, Москва – анкета",
                        "media": [{"id": "photo-41", "type": "photo"}],
                    },
                ]
            },
            ensure_ascii=False,
        )

    async def media(_payload: dict[str, Any]) -> str:
        raise AssertionError("media processing must not start in stop mode")

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
        )
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=None,
        vision_describer=FakeTerminalVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(
        user_request=json.dumps(
            {"chat_id": "daivinchik-chat", "attention_mode": "stop"}
        )
    )

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    assert capsule.summary == "Daivinchik attention required."
    assert "identity_verification" in capsule.markdown_report
    assert not (tmp_path / "social" / "daivinchik" / "taste_profile.md").exists()
    cases_path = tmp_path / "social" / "daivinchik" / "attention_cases.jsonl"
    assert "identity_verification" in cases_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_media_error_degrades_to_text_only_and_still_reports(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "messages": [
                    {
                        "id": 20,
                        "text": "Лена, 25\nКазань\nсырой приватный текст анкеты",
                        "media": [{"id": "photo-20", "type": "photo"}],
                    },
                    {"id": 21, "text": "❤️"},
                ]
            },
            ensure_ascii=False,
        )

    async def media(_payload: dict[str, Any]) -> str:
        raise ValueError("download failed")

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                media,
            ),
        )
    )
    vision = FakeVision()
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=vision,
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(user_request=json.dumps({"chat_id": "daivinchik-chat"}))

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    report = (tmp_path / "social" / "daivinchik" / "taste_profile.md").read_text(
        encoding="utf-8"
    )
    assert capsule.summary == "Daivinchik taste profile completed."
    assert "Ошибок media: 1" in capsule.markdown_report
    assert "text-only" in report
    assert vision.calls == []
    assert "сырой приватный текст анкеты" not in report


@pytest.mark.asyncio
async def test_too_small_history_produces_honest_private_markdown(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.workers.daivinchik_profile import (
        DaivinchikTasteProfileWorkerBackend,
    )

    async def read(_payload: dict[str, Any]) -> str:
        return json.dumps({"messages": []})

    gateway = ToolGateway(
        tools=(FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", read),)
    )
    worker = DaivinchikTasteProfileWorkerBackend(
        tool_gateway=gateway,
        workspace_root=tmp_path,
        llm=FakeVision(),
        frame_extractor=FakeFrameExtractor(),
    )
    pack = ContextPack(user_request=json.dumps({"chat_id": "daivinchik-chat"}))

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    report = (tmp_path / "social" / "daivinchik" / "taste_profile.md").read_text(
        encoding="utf-8"
    )
    assert "данных недостаточно" in report.lower()
    assert "Прочитано сообщений: 0" in capsule.markdown_report
    assert "Доля уверенных выводов: 0%" in capsule.markdown_report
