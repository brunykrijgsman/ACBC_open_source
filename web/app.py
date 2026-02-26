"""
Web frontend for the ACBC survey engine.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from acbc.engine import ACBCEngine
from acbc.models import (
    BYOQuestion,
    ChoiceQuestion,
    MustHaveQuestion,
    ScreeningQuestion,
    SurveyConfig,
    UnacceptableQuestion,
)
from acbc.io import save_raw_results


WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_config: SurveyConfig | None = None
_seed: int | None = None
_output_dir: Path = Path("data")

SESSION_COOKIE = "acbc_session"
sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _next_participant_id() -> str:
    raw_dir = _output_dir / "raw"
    highest = 0
    if raw_dir.is_dir():
        for f in raw_dir.glob("*.json"):
            prefix = f.stem.split("_")[0]
            if prefix.startswith("P") and prefix[1:].isdigit():
                highest = max(highest, int(prefix[1:]))
    return f"P{highest + 1:03d}"


def _question_progress(session: dict[str, Any], engine: ACBCEngine) -> int:
    try:
        state = engine.state
        idx = getattr(state, "question_index", None)
        total = getattr(state, "total_questions", None)

        if idx is None:
            idx = getattr(state, "step", None)
            total = total or getattr(state, "n_steps", None)

        if idx is None:
            idx = getattr(state, "t", None)
            total = total or getattr(state, "T", None)

        if idx is not None and total:
            idx_int = int(idx)
            total_int = max(1, int(total))
            return int(min(idx_int, total_int) / total_int * 100)
    except Exception:
        pass

    answered = int(session.get("answered", 0))
    n_attr = len(engine.config.attributes)
    estimated_total = max(1, n_attr + 1 + 1 + 1 + 8)
    raw = int(answered / estimated_total * 100)
    return min(raw, 95)


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------

def create_app(
    config: SurveyConfig,
    *,
    seed: int | None = None,
    output_dir: Path = Path("data"),
) -> FastAPI:

    global _config, _seed, _output_dir
    _config = config
    _seed = seed
    _output_dir = output_dir

    app = FastAPI(title="ACBC Survey")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --------------------------------------------------
    # Routes
    # --------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def welcome(request: Request):
        return templates.TemplateResponse(
            "welcome.html",
            {"request": request, "config": _config},
        )

    @app.post("/start")
    async def start(request: Request):
        engine = ACBCEngine(_config, seed=_seed)
        pid = _next_participant_id()
        sid = uuid.uuid4().hex

        sessions[sid] = {
            "engine": engine,
            "participant_id": pid,
            "answered": 0,
        }

        response = RedirectResponse(url="/question", status_code=303)
        response.set_cookie(SESSION_COOKIE, sid)
        return response

    @app.get("/question", response_class=HTMLResponse)
    async def question_page(request: Request):
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            return RedirectResponse(url="/")

        session = sessions[sid]
        engine: ACBCEngine = session["engine"]

        if engine.is_complete:
            return RedirectResponse(url="/complete")

        q = engine.get_current_question()
        progress = _question_progress(session, engine)

        ctx: dict[str, Any] = {
            "request": request,
            "question": q,
            "progress": progress,
            "participant_id": session["participant_id"],
        }

        if isinstance(q, BYOQuestion):
            a = q.attribute
            ctx["attr_def"] = (
                getattr(a, "definition", None)
                or getattr(a, "description", None)
                or getattr(a, "help", None)
                or getattr(a, "prompt", None)
                or ""
            )
            return templates.TemplateResponse("byo.html", ctx)

        if isinstance(q, ScreeningQuestion):
            attrs = engine.config.attributes
            ctx["attr_names"] = [a.name for a in attrs]
            ctx["attr_defs"] = {a.name: (a.definition or "") for a in attrs}
            return templates.TemplateResponse("screening.html", ctx)

        if isinstance(q, UnacceptableQuestion):
            ctx["rule_type"] = "unacceptable"
            return templates.TemplateResponse("rule_check.html", ctx)

        if isinstance(q, MustHaveQuestion):
            ctx["rule_type"] = "must_have"
            return templates.TemplateResponse("rule_check.html", ctx)

        if isinstance(q, ChoiceQuestion):
            if not q.scenarios:
                return RedirectResponse(url="/complete")
            attrs = engine.config.attributes
            ctx["attr_names"] = [a.name for a in attrs]
            ctx["attr_defs"] = {a.name: (a.definition or "") for a in attrs}
            return templates.TemplateResponse("choice.html", ctx)

        return RedirectResponse(url="/")

    @app.post("/answer")
    async def submit_answer(request: Request):
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            return RedirectResponse(url="/")

        session = sessions[sid]
        engine: ACBCEngine = session["engine"]
        form = await request.form()

        q = engine.get_current_question()
        did_submit = False

        if isinstance(q, BYOQuestion):
            answer = form.get("level")
            if answer:
                engine.submit_answer(answer)
                did_submit = True

        elif isinstance(q, ScreeningQuestion):
            responses = {}
            for i in range(len(q.scenarios)):
                responses[i] = form.get(f"scenario_{i}") == "accept"
            engine.submit_answer(responses)
            did_submit = True

        elif isinstance(q, (UnacceptableQuestion, MustHaveQuestion)):
            engine.submit_answer(form.get("confirmed") == "yes")
            did_submit = True

        elif isinstance(q, ChoiceQuestion):
            chosen = form.get("chosen")
            if chosen is not None:
                engine.submit_answer(int(chosen))
                did_submit = True

        if did_submit:
            session["answered"] += 1

        if engine.is_complete:
            results = engine.get_results()
            save_raw_results(
                results,
                session["participant_id"],
                _output_dir,
                seed=_seed,
            )
            return RedirectResponse(url="/complete", status_code=303)

        return RedirectResponse(url="/question", status_code=303)

    @app.get("/complete", response_class=HTMLResponse)
    async def complete(request: Request):
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            return RedirectResponse(url="/")

        session = sessions[sid]
        engine: ACBCEngine = session["engine"]

        return templates.TemplateResponse(
            "complete.html",
            {
                "request": request,
                "participant_id": session["participant_id"],
                "winner": engine.state.winner,
                "attr_names": [a.name for a in engine.config.attributes],
                "output_dir": str(_output_dir),
                "progress": 100,
            },
        )

    return app