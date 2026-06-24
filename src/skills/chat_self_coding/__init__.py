"""Chat-mode UX layer for self-coding pipeline (Phase 40).

Wraps the slash-command-driven self-coding pipeline (``ideation_to_spec`` +
``implement_spec`` + ``spec_command``) in a conversational mode meant for
the project owner — Никита — acting as the orchestrator/director rather
than the implementer. ``/код`` or ``/code`` enters the mode; the natural word
"выход" leaves it. Inside the mode the user talks normally; the skill
classifies intent (create / show / approve / reject / run / exit /
status / other) and routes to the underlying pipeline.

Telegram files/photos sent while the room is open are saved as raw
workspace artifacts and referenced from the discussion context, so later
spec creation can hand Architect/Editor the original paths.

Ordinary discussion can use a read-only Codex Explorer before any spec is
created. That lets Жвуша inspect repository code, tests, logs and saved
attachments while still staying in dialogue mode: no file edits, no commits.
Spec creation starts only after an explicit plan/spec request inside /код or
after a short implementation confirmation that follows a durable engineering
proposal from the normal chat.

Editor and Architect cycles emit per-stage block events through a
``BlockPublisher`` (Redis Pub/Sub). A listener in the bot process picks
them up and renders them as four short Telegram messages — 📋 План →
🔧 Подготовка → ✏️ Реализация → ✅ Готово — translated from the
technical audit log into architectural-level prose.
"""
