import base64
import html
import io
import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from anthropic import Anthropic, APIError, AuthenticationError, RateLimitError
from PIL import Image, ImageOps

APP_VERSION = "0.1"


def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


MODEL = secret("MODEL", "claude-sonnet-5-5")

STATUS = {
    "met": {"label": "충족", "color": "#C9F0A8"},
    "partial": {"label": "부분 충족", "color": "#FFEB99"},
    "missing": {"label": "미충족", "color": "#FFD3DE"},
    "unsure": {"label": "확인 필요", "color": "#E2E5EB"},
}
SCORE_VALUE = {"met": 1.0, "partial": 0.5, "missing": 0.0}
TEACHER_OPTIONS = ["아직 모름", "충족", "부분 충족", "미충족"]
TEACHER_TO_STATUS = {"충족": "met", "부분 충족": "partial", "미충족": "missing"}

st.set_page_config(page_title="기준표 점검기", page_icon="📝", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600;700&display=swap');
h1, h2, h3, p, li, label, textarea, input {
  font-family: 'IBM Plex Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
}
h1 { letter-spacing: -0.02em; }
.lead { color: #4A5162; font-size: 1.02rem; line-height: 1.7; margin-top: -0.4rem; }
.draft {
  line-height: 2.1; font-size: 1.02rem; padding: 1rem 1.1rem;
  border: 1px solid #D9DDE5; border-radius: 6px; background: #FFFFFF; color: #1F2430;
}
.draft mark { padding: 0.12em 0.15em; border-radius: 3px; color: #1F2430; }
.draft sup.tag { font-size: 0.7rem; font-weight: 700; color: #2F5BEA; margin-left: 1px; }
.badge {
  display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px;
  font-size: 0.85rem; font-weight: 600; color: #1F2430; margin-left: 0.3rem;
}
.legend { font-size: 0.9rem; color: #4A5162; margin: 0.4rem 0 0.8rem; }
.legend .badge { margin: 0 0.4rem 0 0; }
.quote { border-left: 3px solid #D9DDE5; padding-left: 0.7rem; color: #4A5162; margin: 0.35rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


def password_gate():
    pw = secret("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    st.title("기준표 점검기")
    entered = st.text_input("접속 코드", type="password")
    if st.button("들어가기", type="primary"):
        if entered == pw:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("접속 코드가 맞지 않아요. 다시 입력해 주세요.")
    st.stop()


@st.cache_resource
def get_client():
    key = secret("ANTHROPIC_API_KEY")
    return Anthropic(api_key=key) if key else None


RUBRIC_TOOL = {
    "name": "save_rubric",
    "description": "채점 기준표에서 읽어낸 평가 항목을 저장한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "평가 항목 이름"},
                        "criteria": {"type": "string", "description": "가장 높은 등급(만점)의 기준, 원문 표현 그대로"},
                        "levels": {"type": "string", "description": "나머지 등급별 기준. 없으면 빈 문자열"},
                        "points": {"type": ["number", "null"], "description": "배점. 적혀 있지 않으면 null"},
                    },
                    "required": ["name", "criteria", "levels", "points"],
                },
            },
            "unreadable": {
                "type": "array",
                "items": {"type": "string"},
                "description": "읽기 어려웠거나 확실하지 않은 부분",
            },
        },
        "required": ["items", "unreadable"],
    },
}

TRANSCRIBE_TOOL = {
    "name": "save_text",
    "description": "사진 속 글을 그대로 옮겨 저장한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "unreadable_count": {"type": "integer", "description": "[?]로 표시한 읽기 어려운 부분의 개수"},
        },
        "required": ["text", "unreadable_count"],
    },
}

CHECK_TOOL = {
    "name": "save_check",
    "description": "기준 항목별 점검 결과를 저장한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_no": {"type": "integer", "description": "기준 항목 번호 (1부터)"},
                        "status": {"type": "string", "enum": ["met", "partial", "missing"]},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "근거가 되는 초안 문장 번호 (1부터). 없으면 빈 배열",
                        },
                        "reason": {"type": "string", "description": "판단 이유, 쉬운 말로 1~2문장"},
                        "check_question": {
                            "type": "string",
                            "description": "학생이 스스로 보완하도록 돕는 질문 1개. 고쳐 쓴 문장이나 예시 문장 금지",
                        },
                    },
                    "required": ["item_no", "status", "evidence", "reason", "check_question"],
                },
            }
        },
        "required": ["results"],
    },
}

RUBRIC_SYSTEM = """너는 학교 수행평가 '채점 기준표'를 읽어 평가 항목을 정리하는 역할만 한다.
규칙:
- 기준표에 실제로 적힌 내용만 옮긴다. 없는 항목이나 배점을 지어내지 않는다.
- 항목 이름과 기준 설명은 원문 표현을 최대한 그대로 쓴다.
- 등급별 기준(상/중/하, A/B/C 등)이 있으면 가장 높은 등급의 기준을 criteria에, 나머지 등급 설명을 levels에 옮긴다.
- 배점이 적혀 있지 않으면 points는 null로 둔다.
- 읽을 수 없거나 확실하지 않은 부분은 unreadable에 적는다.
- 채점 기준표가 아닌 이미지나 글이면 items를 빈 배열로 두고 unreadable에 그 이유를 적는다."""

TRANSCRIBE_SYSTEM = """너는 사진 속 학생 글을 있는 그대로 옮겨 적는 역할만 한다.
규칙:
- 맞춤법, 띄어쓰기, 문장을 고치지 않는다. 내용을 더하거나 빼지 않는다.
- 문단 구분은 빈 줄로 유지한다.
- 읽을 수 없는 글자나 단어는 [?]로 표시한다."""

CHECK_SYSTEM = """너는 학생의 수행평가 초안이 채점 기준표의 각 항목을 충족하는지 '점검'만 한다.
절대 규칙:
- 학생 대신 글을 쓰지 않는다. 고쳐 쓴 문장, 예시 문장, 덧붙일 내용의 초안을 절대 제시하지 않는다.
- 근거는 반드시 제공된 문장 번호로만 댄다. 초안에 없는 내용을 근거로 삼지 않는다.
- 근거가 되는 문장이 없으면 status는 missing, evidence는 빈 배열로 둔다.
- 기준을 일부만 만족하면 partial로 둔다. 애매하면 더 엄격하게 판단한다.
- reason은 고등학생이 이해하기 쉬운 말로 1~2문장으로 쓴다.
- check_question은 학생이 스스로 초안을 다시 살펴보게 하는 질문 1개다. 질문 안에 정답 문장을 넣지 않는다.
  met인 항목은 더 단단하게 만들 점을 묻는 질문으로 쓴다.
- 모든 기준 항목에 대해 결과를 하나씩 낸다."""

EXPLAIN_SYSTEM = """너는 수행평가 채점 기준 한 항목의 '뜻'을 고등학생에게 쉬운 말로 풀어주는 역할만 한다.
규칙:
- 2~4문장으로, 선생님이 이 기준으로 무엇을 확인하려는지 설명한다.
- 일반적으로 어떤 요소(예: 출처 밝히기, 근거 제시, 자기 생각 구분)가 필요한지는 말해도 된다.
- 학생의 주제에 맞춘 예시 문장이나 글의 일부를 대신 써주지 않는다.
- 확실하지 않으면 선생님께 확인해 보라고 말한다."""


def image_block(uploaded):
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def call_tool(system, content, tool, max_tokens=4000):
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": content}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("AI 응답에서 결과를 찾지 못했어요.")


def call_text(system, prompt, max_tokens=600):
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def run_ai(spinner_text, fn, *args):
    try:
        with st.spinner(spinner_text):
            return fn(*args)
    except AuthenticationError:
        st.error("API 키가 올바르지 않아요. Streamlit 앱 설정의 Secrets를 확인해 주세요.")
    except RateLimitError:
        st.error("요청이 잠시 몰렸어요. 1분쯤 뒤에 다시 눌러 주세요.")
    except APIError as e:
        st.error(f"AI 서버와 통신하지 못했어요. 잠시 뒤 다시 시도해 주세요. ({type(e).__name__})")
    except Exception as e:
        st.error(f"처리 중 문제가 생겼어요: {e}")
    return None


def split_sentences(text):
    text = text.replace("\r", "")
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def validate_results(raw_results, n_items, n_sentences):
    by_item = {}
    for r in raw_results or []:
        no = r.get("item_no")
        if not isinstance(no, int) or not (1 <= no <= n_items) or (no - 1) in by_item:
            continue
        flags = []
        evidence_raw = r.get("evidence") or []
        evidence = sorted({e - 1 for e in evidence_raw if isinstance(e, int) and 1 <= e <= n_sentences})
        if len(evidence) < len(set(evidence_raw)):
            flags.append("AI가 존재하지 않는 문장 번호를 근거로 들어 그 부분은 제외했어요.")
        status = r.get("status")
        if status not in SCORE_VALUE:
            status = "unsure"
            flags.append("판정 형식이 올바르지 않아 '확인 필요'로 표시했어요.")
        if status in ("met", "partial") and not evidence:
            status = "unsure"
            flags.append("충족이라고 판단했지만 근거 문장이 없어 '확인 필요'로 바꿨어요.")
        if status == "missing" and evidence:
            evidence = []
        by_item[no - 1] = {
            "status": status,
            "evidence": evidence,
            "reason": (r.get("reason") or "").strip(),
            "check_question": (r.get("check_question") or "").strip(),
            "flags": flags,
        }
    for i in range(n_items):
        if i not in by_item:
            by_item[i] = {
                "status": "unsure",
                "evidence": [],
                "reason": "AI가 이 항목을 판단하지 못했어요.",
                "check_question": "이 기준을 초안의 어디에서 채웠는지 직접 찾아볼까요?",
                "flags": ["AI 결과에 이 항목이 빠져 있었어요."],
            }
    return [by_item[i] for i in range(n_items)]


def expected_ratio(items, results):
    use_points = all(isinstance(it.get("points"), (int, float)) and it["points"] > 0 for it in items)
    total = got = 0.0
    for it, r in zip(items, results):
        if r["status"] not in SCORE_VALUE:
            continue
        w = it["points"] if use_points else 1.0
        total += w
        got += w * SCORE_VALUE[r["status"]]
    return (got / total if total else None), use_points


def clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def badge(status):
    s = STATUS[status]
    return f'<span class="badge" style="background:{s["color"]}">{s["label"]}</span>'


def legend():
    st.markdown(
        '<div class="legend">' + "".join(badge(k) for k in STATUS) + "</div>",
        unsafe_allow_html=True,
    )


def draft_html(sentences, results):
    rank = {"met": 3, "partial": 2, "unsure": 1}
    marks = {}
    for i, r in enumerate(results):
        for s in r["evidence"]:
            cur, nums = marks.get(s, (None, []))
            nums.append(i + 1)
            if cur is None or rank.get(r["status"], 0) > rank.get(cur, 0):
                cur = r["status"]
            marks[s] = (cur, nums)
    parts = []
    for idx, s in enumerate(sentences):
        text = html.escape(s)
        if idx in marks:
            status, nums = marks[idx]
            tags = "".join(f'<sup class="tag">{n}</sup>' for n in sorted(set(nums)))
            parts.append(f'<mark style="background:{STATUS[status]["color"]}">{text}</mark>{tags}')
        else:
            parts.append(f"<span>{text}</span>")
    return '<div class="draft">' + " ".join(parts) + "</div>"


def reset_all():
    for k in list(st.session_state.keys()):
        if k != "authed":
            del st.session_state[k]


def init_state():
    defaults = {
        "rubric_items": None,
        "rubric_unreadable": [],
        "rubric_confirmed": False,
        "draft_text": "",
        "sentences": [],
        "results": None,
        "explanations": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def step_rubric():
    st.subheader("1. 채점 기준표 올리기")
    method = st.radio("기준표를 어떻게 넣을까요?", ["사진 올리기", "사진 찍기", "글로 붙여넣기"], horizontal=True)
    content = None
    if method == "사진 올리기":
        files = st.file_uploader(
            "기준표 사진 (여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
        )
        if files:
            content = files
    elif method == "사진 찍기":
        shot = st.camera_input("기준표가 화면에 꽉 차게 찍어 주세요")
        if shot:
            content = [shot]
    else:
        text = st.text_area("기준표 내용을 붙여넣어 주세요", height=180)
        if text.strip():
            content = text

    if st.button("기준표 읽기", type="primary", disabled=content is None):
        if isinstance(content, str):
            payload = [{"type": "text", "text": f"다음은 채점 기준표다.\n\n{content}"}]
        else:
            try:
                payload = [image_block(f) for f in content]
            except Exception:
                st.error("사진 파일을 열지 못했어요. jpg나 png 사진으로 다시 올려 주세요.")
                return
            payload.append({"type": "text", "text": "이 사진 속 채점 기준표를 정리해 줘."})
        out = run_ai("기준표를 읽고 있어요…", call_tool, RUBRIC_SYSTEM, payload, RUBRIC_TOOL)
        if out is not None:
            st.session_state.rubric_items = out.get("items") or []
            st.session_state.rubric_unreadable = out.get("unreadable") or []
            st.session_state.rubric_confirmed = False
            st.session_state.results = None

    items = st.session_state.rubric_items
    if items is None:
        return
    if not items:
        st.warning("기준표 항목을 찾지 못했어요. 채점 기준표가 잘 보이게 다시 찍거나 글로 붙여넣어 주세요.")
        for u in st.session_state.rubric_unreadable:
            st.caption(f"· {u}")
        return

    st.markdown("AI가 읽은 내용이 맞는지 확인해 주세요. 틀린 칸은 직접 고치고, 빠진 항목은 아래 빈 줄에 추가하면 돼요.")
    for u in st.session_state.rubric_unreadable:
        st.warning(f"잘 안 읽힌 부분: {u}")
    df = pd.DataFrame(items, columns=["name", "criteria", "levels", "points"])
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        key="rubric_editor",
        column_config={
            "name": st.column_config.TextColumn("평가 항목", required=True),
            "criteria": st.column_config.TextColumn("만점 기준", width="large"),
            "levels": st.column_config.TextColumn("등급별 기준"),
            "points": st.column_config.NumberColumn("배점", min_value=0),
        },
    )
    if st.button("이 기준표로 확정"):
        confirmed = []
        for _, row in edited.iterrows():
            name = clean(row.get("name"))
            if not name:
                continue
            pts = row.get("points")
            confirmed.append(
                {
                    "name": name,
                    "criteria": clean(row.get("criteria")),
                    "levels": clean(row.get("levels")),
                    "points": float(pts) if pd.notna(pts) else None,
                }
            )
        if not confirmed:
            st.error("항목이 하나도 없어요. 평가 항목을 한 개 이상 입력해 주세요.")
        else:
            st.session_state.rubric_items = confirmed
            st.session_state.rubric_confirmed = True
            st.session_state.results = None
            st.success(f"기준 항목 {len(confirmed)}개를 확정했어요.")


def step_draft():
    st.subheader("2. 내 초안 넣기")
    if not st.session_state.rubric_confirmed:
        st.info("먼저 위에서 기준표를 확정해 주세요.")
        return False
    method = st.radio("초안을 어떻게 넣을까요?", ["글로 붙여넣기", "사진 올리기"], horizontal=True, key="draft_method")
    if method == "사진 올리기":
        files = st.file_uploader(
            "초안 사진 (여러 장이면 순서대로)", type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True, key="draft_files",
        )
        if files and st.button("사진 속 글 옮기기"):
            try:
                payload = [image_block(f) for f in files]
            except Exception:
                st.error("사진 파일을 열지 못했어요. jpg나 png 사진으로 다시 올려 주세요.")
                return False
            payload.append({"type": "text", "text": "사진 속 글을 순서대로 그대로 옮겨 줘."})
            out = run_ai("사진 속 글을 옮기고 있어요…", call_tool, TRANSCRIBE_SYSTEM, payload, TRANSCRIBE_TOOL)
            if out is not None:
                st.session_state.draft_text = out.get("text", "")
                if out.get("unreadable_count"):
                    st.session_state.draft_notice = f"읽기 어려운 부분 {out['unreadable_count']}곳을 [?]로 표시했어요."
                st.rerun()
        if st.session_state.get("draft_notice"):
            st.warning(st.session_state.draft_notice + " 아래 글을 원본과 비교해서 고쳐 주세요.")
    st.text_area(
        "초안 (점검 전에 내용이 맞는지 확인해 주세요)",
        key="draft_text",
        height=260,
        placeholder="수행평가 초안을 붙여넣어 주세요. 이름이나 학번은 지우고 넣어 주세요.",
    )
    return True


def step_check():
    st.subheader("3. 점검 결과")
    draft = st.session_state.draft_text.strip()
    items = st.session_state.rubric_items
    if st.button("점검하기", type="primary", disabled=not draft):
        if len(draft) < 30:
            st.error("초안이 너무 짧아요. 최소 두세 문장은 넣어 주세요.")
            return
        if len(draft) > 15000:
            st.error("초안이 너무 길어요. 15,000자 이하로 나눠서 점검해 주세요.")
            return
        sentences = split_sentences(draft)
        rubric_txt = "\n".join(
            f"{i + 1}. {it['name']}"
            + (f" (배점 {it['points']:g})" if it.get("points") is not None else "")
            + f"\n   만점 기준: {it['criteria']}"
            + (f"\n   등급별 기준: {it['levels']}" if it.get("levels") else "")
            for i, it in enumerate(items)
        )
        sent_txt = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sentences))
        prompt = f"[채점 기준표]\n{rubric_txt}\n\n[초안 문장]\n{sent_txt}"
        out = run_ai("기준 항목과 초안을 맞춰 보고 있어요…", call_tool, CHECK_SYSTEM, prompt, CHECK_TOOL, 6000)
        if out is not None:
            st.session_state.sentences = sentences
            st.session_state.results = validate_results(out.get("results"), len(items), len(sentences))
            st.session_state.explanations = {}
            st.session_state.checked_at = datetime.now().isoformat(timespec="seconds")
            for k in [k for k in st.session_state.keys() if str(k).startswith(("teacher_", "todo_"))]:
                del st.session_state[k]

    results = st.session_state.results
    if not results:
        st.caption("기준표와 초안을 넣고 '점검하기'를 누르면 결과가 여기에 나와요.")
        return

    counts = {k: sum(r["status"] == k for r in results) for k in STATUS}
    ratio, use_points = expected_ratio(items, results)
    cols = st.columns(4)
    for col, k in zip(cols, STATUS):
        col.metric(STATUS[k]["label"], f"{counts[k]}개")
    if ratio is not None:
        basis = "배점 기준" if use_points else "항목 수 기준"
        st.progress(ratio, text=f"예상 충족도 {ratio * 100:.0f}% ({basis}, 선생님 점수가 아니에요)")

    st.markdown("**초안에서 근거로 쓰인 문장**")
    st.caption("문장 옆 작은 파란 숫자는 기준 항목 번호예요.")
    legend()
    st.markdown(draft_html(st.session_state.sentences, results), unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.markdown("**항목별 결과**")
    for i, (it, r) in enumerate(zip(items, results)):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {html.escape(it['name'])}** {badge(r['status'])}", unsafe_allow_html=True)
            if it.get("criteria"):
                st.caption(f"만점 기준: {it['criteria']}")
            if r["reason"]:
                st.write(r["reason"])
            for s in r["evidence"]:
                st.markdown(
                    f'<div class="quote">[{s + 1}] {html.escape(st.session_state.sentences[s])}</div>',
                    unsafe_allow_html=True,
                )
            for f in r["flags"]:
                st.caption(f"⚠ {f}")
            if r["check_question"]:
                st.markdown(f"스스로 점검해 보기: *{html.escape(r['check_question'])}*")
            if st.button("이 기준은 무슨 뜻이에요?", key=f"explain_{i}"):
                prompt = f"평가 항목: {it['name']}\n만점 기준: {it['criteria']}\n등급별 기준: {it.get('levels') or '없음'}"
                text = run_ai("기준의 뜻을 풀어보고 있어요…", call_text, EXPLAIN_SYSTEM, prompt)
                if text:
                    st.session_state.explanations[i] = text
            if i in st.session_state.explanations:
                st.info(st.session_state.explanations[i])

    todo = [(i, r) for i, r in enumerate(results) if r["status"] != "met" and r["check_question"]]
    if todo:
        st.markdown("**보완 체크리스트**")
        st.caption("초안을 고친 뒤 하나씩 체크해 보세요. 체크 표시는 이 화면에서만 유지돼요.")
        for i, r in todo:
            st.checkbox(f"{i + 1}번: {r['check_question']}", key=f"todo_{i}")


def tab_teacher():
    results = st.session_state.results
    items = st.session_state.rubric_items
    if not results:
        st.info("먼저 '점검하기' 탭에서 점검을 끝내 주세요. 수행평가가 채점된 뒤 여기에서 결과를 비교해요.")
        return
    st.markdown("선생님께 받은 채점 결과를 항목별로 골라 주세요. 서비스 판단이 얼마나 맞았는지 계산해요.")
    labels = []
    for i, (it, r) in enumerate(zip(items, results)):
        c1, c2 = st.columns([3, 4])
        c1.markdown(f"**{i + 1}. {html.escape(it['name'])}**<br>서비스 판단 {badge(r['status'])}", unsafe_allow_html=True)
        choice = c2.radio("선생님 채점", TEACHER_OPTIONS, horizontal=True, key=f"teacher_{i}", label_visibility="collapsed")
        labels.append(TEACHER_TO_STATUS.get(choice))

    judged = [(r["status"], t) for r, t in zip(results, labels) if t]
    if judged:
        hit = sum(a == t for a, t in judged)
        st.metric("일치율", f"{hit / len(judged) * 100:.0f}%", f"{hit}/{len(judged)} 항목 일치", delta_color="off")
    teacher_score = st.text_input("선생님이 주신 총점 (선택)", placeholder="예: 18/20")
    teacher_note = st.text_area("선생님 피드백 메모 (선택)", height=80)

    include_draft = st.checkbox("초안 내용도 기록에 포함하기", value=False)
    record = {
        "app_version": APP_VERSION,
        "model": MODEL,
        "checked_at": st.session_state.get("checked_at"),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "rubric": items,
        "results": [
            {"item": it["name"], "ai_status": r["status"], "evidence": [e + 1 for e in r["evidence"]],
             "reason": r["reason"], "flags": r["flags"], "teacher_status": t}
            for it, r, t in zip(items, results, labels)
        ],
        "teacher_score": teacher_score,
        "teacher_note": teacher_note,
        "sentences": st.session_state.sentences if include_draft else None,
    }
    st.download_button(
        "기록 내보내기 (JSON)",
        data=json.dumps(record, ensure_ascii=False, indent=2),
        file_name=f"check_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )
    st.caption("이 사이트는 기록을 저장하지 않아요. 내보낸 파일을 모아 '평가 기록' 탭에 올리면 정확도를 볼 수 있어요.")


def tab_eval():
    st.markdown("내보낸 기록 파일을 여러 개 올리면 서비스 판단과 선생님 채점의 일치율을 모아서 보여줘요.")
    files = st.file_uploader("기록 파일 (JSON)", type=["json"], accept_multiple_files=True, key="eval_files")
    if not files:
        st.caption("아직 올린 기록이 없어요. '선생님 채점과 비교' 탭에서 기록을 내보낸 뒤 여기에 올려 주세요.")
        return
    rows, bad = [], []
    for f in files:
        try:
            rec = json.load(f)
            for r in rec.get("results", []):
                if r.get("teacher_status"):
                    rows.append({
                        "파일": f.name, "버전": rec.get("app_version", "?"), "항목": r.get("item"),
                        "서비스": STATUS.get(r.get("ai_status"), {"label": "?"})["label"],
                        "선생님": STATUS.get(r.get("teacher_status"), {"label": "?"})["label"],
                        "일치": r.get("ai_status") == r.get("teacher_status"),
                        "판단 이유": r.get("reason", ""),
                    })
        except Exception:
            bad.append(f.name)
    for b in bad:
        st.warning(f"{b} 파일은 읽을 수 없어 제외했어요.")
    if not rows:
        st.info("선생님 채점이 입력된 항목이 없어요.")
        return
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("기록 수", f"{df['파일'].nunique()}건")
    c2.metric("비교한 항목", f"{len(df)}개")
    c3.metric("전체 일치율", f"{df['일치'].mean() * 100:.0f}%")

    st.markdown("**버전별 일치율**")
    st.caption("프롬프트나 규칙을 고친 뒤 나아졌는지 확인해요.")
    by_ver = df.groupby("버전")["일치"].agg(["mean", "count"]).reset_index()
    by_ver["일치율(%)"] = (by_ver["mean"] * 100).round(0)
    st.dataframe(by_ver[["버전", "일치율(%)", "count"]].rename(columns={"count": "항목 수"}), hide_index=True)

    st.markdown("**서비스 판단과 선생님 채점 비교표**")
    st.caption("행은 서비스 판단, 열은 선생님 채점이에요. 같은 이름이 만나는 칸 밖의 숫자가 틀린 판단이에요.")
    st.dataframe(pd.crosstab(df["서비스"], df["선생님"]))

    st.markdown("**틀린 판단 모아보기**")
    st.caption("개선 기록의 재료가 돼요.")
    st.dataframe(df[~df["일치"]][["파일", "항목", "서비스", "선생님", "판단 이유"]], hide_index=True)


password_gate()
init_state()

st.title("기준표 점검기")
st.markdown(
    '<p class="lead">수행평가 채점 기준표와 내 초안을 맞대어, 어떤 기준을 채웠고 무엇이 빠졌는지 보여줘요. '
    "글을 대신 써주지는 않아요. 고치는 건 내 몫이에요.</p>",
    unsafe_allow_html=True,
)

if get_client() is None:
    st.error("API 키가 설정되지 않았어요. Streamlit 앱 설정의 Secrets에 ANTHROPIC_API_KEY를 넣어 주세요.")
    st.stop()

with st.sidebar:
    st.markdown("**기준표 점검기** v" + APP_VERSION)
    st.caption("올린 기준표와 초안은 분석을 위해 AI(Anthropic)로 전송되고, 이 사이트에는 저장되지 않아요. 이름·학번은 지우고 올려 주세요.")
    st.caption("결과는 참고용이에요. 최종 판단은 선생님 기준을 따르세요.")
    if st.button("처음부터 다시"):
        reset_all()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["점검하기", "선생님 채점과 비교", "평가 기록"])
with tab1:
    step_rubric()
    st.divider()
    if step_draft():
        st.divider()
        step_check()
with tab2:
    tab_teacher()
with tab3:
    tab_eval()
