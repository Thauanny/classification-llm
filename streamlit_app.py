"""
Aplicação Streamlit — Interface gráfica para classificação com Ollama.

Funcionalidades:
- Classificação de texto único
- Classificação de dataset (CSV / Excel) com seleção de coluna
- Progresso real de download do modelo
- Remoção de modelos instalados
- Configuração completa de parâmetros Ollama
- Download dos resultados
"""

import io
import json

import pandas as pd
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Classificador com Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Helpers — sem side-effects, apenas computação
# ──────────────────────────────────────────────

def _get(url: str, timeout: int = 5):
    """GET seguro — retorna Response ou None."""
    try:
        return requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException:
        return None


def fetch_models(base_url: str) -> list[dict]:
    """Retorna lista de modelos disponíveis na API."""
    resp = _get(f"{base_url}/models")
    if resp and resp.status_code == 200:
        return resp.json()
    return []


def check_health(base_url: str) -> dict | None:
    """Retorna payload do /health ou None se offline."""
    resp = _get(f"{base_url}/health")
    if resp and resp.status_code == 200:
        return resp.json()
    return None


def call_classify_text(
    base_url: str,
    text: str,
    prompt_tmpl: str,
    mdl: str,
    temp: float,
    tp: float,
    tk: int,
    mt: int,
    rp: float,
) -> dict:
    """Chama POST /classify/text (Ollama) e retorna classificação + metadata."""
    payload = {
        "text": text,
        "prompt_template": prompt_tmpl,
        "model_name": mdl,
        "params": {
            "temperature": temp,
            "top_p": tp,
            "top_k": tk,
            "max_tokens": mt if mt > 0 else None,
            "repeat_penalty": rp,
        },
    }
    resp = requests.post(f"{base_url}/classify/text", json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return {"classification": data["classification"], "metadata": data.get("metadata") or {}}


def call_classify_text_groq(
    base_url: str,
    text: str,
    prompt_tmpl: str,
    mdl: str,
    api_key: str,
    temp: float,
    tp: float,
    mt: int,
) -> dict:
    """Chama POST /classify/groq/text e retorna classificação + metadata."""
    payload = {
        "text": text,
        "prompt_template": prompt_tmpl,
        "model_name": mdl,
        "api_key": api_key,
        "params": {
            "temperature": temp,
            "top_p": tp,
            "max_tokens": mt if mt > 0 else None,
        },
    }
    resp = requests.post(f"{base_url}/classify/groq/text", json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return {"classification": data["classification"], "metadata": data.get("metadata") or {}}


def load_dataframe(uploaded) -> pd.DataFrame:
    """Carrega CSV ou Excel a partir do UploadedFile do Streamlit."""
    content = uploaded.read()
    uploaded.seek(0)
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    return pd.read_excel(io.BytesIO(content))


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converte DataFrame para CSV UTF-8 BOM (compatível com Excel)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
defaults = {
    "models": [],
    "api_ok": False,
    "single_result": None,
    "dataset_results": None,
    "confirm_delete": None,
    "provider": "Ollama",
    "groq_api_key": "",
}
for _key, _val in defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurações")

    # ── Provedor ────────────────────────────────
    provider = st.radio(
        "Provedor LLM",
        options=["Ollama", "Groq"],
        horizontal=True,
        help="Ollama = modelos locais | Groq = API cloud gratuita",
    )
    st.session_state.provider = provider

    st.divider()
    st.subheader("🔗 API")
    api_url = st.text_input("URL da API", value="http://localhost:8000")
    if provider == "Groq":
        groq_key = st.text_input(
            "🔑 Chave de API Groq",
            value=st.session_state.groq_api_key,
            type="password",
            help="Obtenha em https://console.groq.com/keys — plano free disponível.",
        )
        st.session_state.groq_api_key = groq_key
    else:
        groq_key = ""
    col_btn, col_status = st.columns([1, 2])
    with col_btn:
        if st.button("Conectar", use_container_width=True):
            health = check_health(api_url)
            if health:
                st.session_state.api_ok = True
                st.session_state.models = fetch_models(api_url)
            else:
                st.session_state.api_ok = False
                st.session_state.models = []
    with col_status:
        if st.session_state.api_ok:
            st.success("Online", icon="✅")
        else:
            st.error("Offline", icon="❌")

    st.divider()

    # ── Modelo ───────────────────────────────
    st.subheader("🤖 Modelo para classificação")

    _GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it",
    ]

    raw_models: list[dict] = st.session_state.models
    model_names = [m["name"] for m in raw_models]

    if provider == "Groq":
        model_name = st.selectbox(
            "Modelo Groq",
            options=_GROQ_MODELS,
            help="Modelos disponíveis no free tier do Groq.",
        )
    elif model_names:
        model_name = st.selectbox(
            "Modelo instalado",
            options=model_names,
            help="Modelos disponíveis localmente no Ollama.",
        )
        sel_info = next((m for m in raw_models if m["name"] == model_name), None)
        if sel_info:
            st.caption(f"Tamanho: **{sel_info.get('size', 'N/A')}**")
    else:
        st.info("Nenhum modelo instalado. Baixe um abaixo.", icon="ℹ️")
        model_name = ""  # será preenchido após download

    # ── Baixar / Remover (apenas Ollama) ─────
    if provider == "Ollama":
        _POPULAR_MODELS = [
            "llama3.2:3b-instruct-fp16",
            "llama3.2:3b",
            "llama3.2:1b",
            "llama3.1:8b",
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b-instruct-q8_0",
            "llama3.1:8b-instruct-fp16",
            "mistral:7b",
            "mistral:7b-instruct",
            "phi3:mini",
            "phi3:medium",
            "gemma2:2b",
            "gemma2:9b",
            "qwen2.5:3b",
            "qwen2.5:7b",
            "deepseek-r1:7b",
            "── digitar manualmente ──",
        ]

        with st.expander("⬇️ Baixar modelo", expanded=not bool(model_names)):
            pull_choice = st.selectbox(
                "Selecione o modelo",
                options=_POPULAR_MODELS,
                help="Lista de modelos populares do Ollama. Escolha ou digite manualmente.",
                key="pull_choice",
            )
            if pull_choice == "── digitar manualmente ──":
                pull_name = st.text_input(
                    "Nome do modelo (ex: llama3.2:3b)",
                    key="pull_name_input",
                )
            else:
                pull_name = pull_choice
            if st.button("Iniciar download", use_container_width=True, type="primary"):
                if not st.session_state.api_ok:
                    st.warning("Conecte à API primeiro.")
                else:
                    pull_bar = st.progress(0, text="Iniciando…")
                    pull_status_txt = st.empty()
                    pull_error = st.empty()
                    last_pct = 0.0

                    try:
                        with requests.post(
                            f"{api_url}/models/pull/stream",
                            json={"model_name": pull_name},
                            stream=True,
                            timeout=None,
                        ) as stream_resp:
                            if stream_resp.status_code != 200:
                                st.error(f"Erro HTTP {stream_resp.status_code}: {stream_resp.text}")
                            else:
                                for raw_line in stream_resp.iter_lines():
                                    if not raw_line:
                                        continue
                                    try:
                                        evt = json.loads(raw_line)
                                    except json.JSONDecodeError:
                                        continue

                                    status_txt = evt.get("status", "")
                                    pct = float(evt.get("percent", 0))
                                    completed = int(evt.get("completed", 0))
                                    total_bytes = int(evt.get("total", 0))

                                    if pct > 0:
                                        last_pct = pct
                                        size_info = (
                                            f"  ({_format_bytes(completed)} / {_format_bytes(total_bytes)})"
                                            if total_bytes else ""
                                        )
                                        pull_bar.progress(
                                            min(pct / 100, 1.0),
                                            text=f"Baixando: **{pct:.1f}%**{size_info}",
                                        )
                                    else:
                                        pull_bar.progress(
                                            min(last_pct / 100, 1.0),
                                            text=f"Aguardando… {last_pct:.1f}%",
                                        )

                                    pull_status_txt.caption(f"Status: `{status_txt}`")

                                    if evt.get("status") == "error":
                                        pull_error.error(evt.get("message", "Erro desconhecido."))
                                        break

                                else:
                                    pull_bar.progress(1.0, text="✅ Download concluído!")
                                    pull_status_txt.empty()
                                    st.success(f"Modelo **{pull_name}** baixado com sucesso!")
                                    st.session_state.models = fetch_models(api_url)
                                    st.session_state.confirm_delete = None

                    except requests.exceptions.RequestException as req_exc:
                        st.error(f"Erro de conexão: {req_exc}")

        if model_names:
            st.divider()
            with st.expander("🗑️ Remover modelo instalado"):
                del_name = st.selectbox(
                    "Selecione o modelo para remover",
                    options=model_names,
                    key="del_model_select",
                )
                st.warning(
                    f"Remove **{del_name}** permanentemente do seu sistema.",
                    icon="⚠️",
                )

                if st.session_state.confirm_delete != del_name:
                    if st.button(
                        f"🗑️ Remover {del_name}",
                        use_container_width=True,
                        key="btn_delete_ask",
                    ):
                        st.session_state.confirm_delete = del_name
                        st.rerun()
                else:
                    st.error("Tem certeza? Esta ação não pode ser desfeita.")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("✅ Sim, remover", use_container_width=True, key="btn_confirm_del"):
                            try:
                                del_resp = requests.delete(
                                    f"{api_url}/models/{del_name}",
                                    timeout=30,
                                )
                                if del_resp.status_code == 200:
                                    st.success(f"Modelo **{del_name}** removido!")
                                    st.session_state.confirm_delete = None
                                    st.session_state.models = fetch_models(api_url)
                                    st.rerun()
                                else:
                                    detail = del_resp.json().get("detail", del_resp.text)
                                    st.error(f"Erro: {detail}")
                            except requests.exceptions.RequestException as req_exc:
                                st.error(f"Erro de conexão: {req_exc}")
                    with col_no:
                        if st.button("❌ Cancelar", use_container_width=True, key="btn_cancel_del"):
                            st.session_state.confirm_delete = None
                            st.rerun()
    elif provider == "Groq":
        st.caption("🔗 [console.groq.com/keys](https://console.groq.com/keys) — crie sua chave free")

    st.divider()

    # ── Prompt ───────────────────────────────
    st.subheader("📋 Prompt de Classificação")
    st.caption("Use **{text}** onde o texto será inserido.")
    prompt_template = st.text_area(
        "Prompt template",
        value=(
            "Classifique o sentimento do texto abaixo como exatamente uma dessas opções: "
            "Positivo, Negativo ou Neutro.\n\n"
            "Texto: {text}\n\n"
            "Responda apenas com a classificação, sem explicações:"
        ),
        height=180,
    )

    st.divider()

    # ── Parâmetros Ollama ────────────────────
    st.subheader("🎛️ Parâmetros do Modelo")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05,
        help="0 = determinístico | 2 = muito aleatório.")
    top_p = st.slider("Top-P (nucleus sampling)", 0.0, 1.0, 0.9, 0.05,
        help="Considera tokens até probabilidade acumulada Top-P.")
    if provider == "Ollama":
        top_k = st.number_input("Top-K", 1, 500, 40, 1,
            help="Tokens candidatos por passo.")
        repeat_penalty = st.slider("Repeat Penalty", 0.0, 3.0, 1.1, 0.05,
            help="> 1.0 reduz repetição de tokens.")
    else:
        top_k = 40
        repeat_penalty = 1.1
        st.caption("ℹ️ Top-K e Repeat Penalty não são suportados pelo Groq.")
    max_tokens = st.number_input("Max Tokens (0 = ilimitado)", 0, 8192, 50, 10,
        help="Limita o tamanho da resposta.")

# ──────────────────────────────────────────────
# Área principal
# ──────────────────────────────────────────────
st.title("🤖 Classificador de Textos com Ollama")
st.caption("Classifique textos individuais ou datasets inteiros usando modelos LLM locais.")

if not st.session_state.api_ok:
    st.warning("API offline — clique em **Conectar** na barra lateral.")

tab_single, tab_dataset = st.tabs(["📝 Texto Único", "📊 Dataset"])

# ──────────────────────────────────────────────
# Tab 1 — Texto Único
# ──────────────────────────────────────────────
with tab_single:
    st.subheader("Classificação de Texto Único")

    text_input = st.text_area(
        "Texto para classificar",
        height=180,
        placeholder="Cole ou digite o texto que deseja classificar...",
    )

    col_run, col_clr = st.columns([2, 1])
    with col_run:
        run_single = st.button(
            "🚀 Classificar",
            type="primary",
            disabled=not st.session_state.api_ok,
            use_container_width=True,
        )
    with col_clr:
        if st.button("🗑️ Limpar", use_container_width=True, key="clr_single"):
            st.session_state.single_result = None

    if run_single:
        if not text_input.strip():
            st.warning("Insira um texto para classificar.")
        elif not prompt_template.strip():
            st.warning("Defina um prompt na barra lateral.")
        elif provider == "Groq" and not groq_key.strip():
            st.warning("Insira a chave de API Groq na barra lateral.")
        else:
            with st.spinner("Classificando…"):
                try:
                    if provider == "Groq":
                        cls_result = call_classify_text_groq(
                            api_url, text_input, prompt_template, model_name,
                            groq_key, temperature, top_p, int(max_tokens),
                        )
                    else:
                        cls_result = call_classify_text(
                            api_url, text_input, prompt_template, model_name,
                            temperature, top_p, int(top_k), int(max_tokens), repeat_penalty,
                        )
                    st.session_state.single_result = {
                        "text": text_input,
                        "classification": cls_result["classification"],
                        "model": model_name,
                        "provider": provider,
                        "prompt": prompt_template,
                        "params": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": int(top_k) if provider == "Ollama" else "N/A",
                            "max_tokens": int(max_tokens),
                            "repeat_penalty": repeat_penalty if provider == "Ollama" else "N/A",
                        },
                        "metadata": cls_result.get("metadata", {}),
                    }
                except requests.exceptions.HTTPError as http_exc:
                    detail = http_exc.response.json().get("detail", str(http_exc))
                    st.error(f"Erro da API: {detail}")
                except requests.exceptions.RequestException as req_exc:
                    st.error(f"Erro de conexão: {req_exc}")

    if st.session_state.single_result:
        res = st.session_state.single_result
        st.success("\u2705 Classificação concluída!")
        st.metric(label="Resultado", value=res["classification"])
        st.caption(f"Modelo: `{res['model']}` — Provedor: **{res.get('provider', 'Ollama')}**")
        with st.expander("Ver prompt e parâmetros utilizados"):
            st.markdown("**Prompt enviado ao modelo:**")
            st.code(res.get("prompt", prompt_template).replace("{text}", res["text"]), language="text")

            st.divider()
            st.markdown("**Parâmetros de geração:**")
            p = res.get("params", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Temperature", p.get("temperature", "—"))
            c2.metric("Top-P", p.get("top_p", "—"))
            c3.metric("Top-K", p.get("top_k", "—"))
            c4.metric("Max Tokens", p.get("max_tokens") or "∞")
            c5.metric("Repeat Penalty", p.get("repeat_penalty", "—"))

            meta = res.get("metadata") or {}
            if meta:
                st.divider()
                _prov = res.get("provider", "Ollama")
                st.markdown(f"**Resposta da API ({_prov}):**")

                # ── Tokens ───────────────────────────────────────────────
                _pt = meta.get("prompt_tokens")
                _ct = meta.get("completion_tokens")
                _tt = meta.get("total_tokens")
                _tps = meta.get("tokens_per_second")

                tok_cols = st.columns(4)
                tok_cols[0].metric("Tokens (prompt)", _pt if _pt is not None else "—")
                tok_cols[1].metric("Tokens (resposta)", _ct if _ct is not None else "—")
                tok_cols[2].metric("Tokens (total)", _tt if _tt is not None else "—")
                tok_cols[3].metric("Velocidade de geração", f"{_tps:.1f} tok/s" if _tps else "—")

                if _prov == "Ollama":
                    # ── Durações Ollama ───────────────────────────────────
                    st.caption(
                        "💻 **Tempos internos de hardware** — medidos diretamente na GPU/CPU local "
                        "(carga do modelo na memória, processamento do prompt, geração token a token)."
                    )
                    dur_cols = st.columns(4)
                    dur_cols[0].metric("Duração total", f"{meta['total_duration_s']} s" if meta.get('total_duration_s') else "—")
                    dur_cols[1].metric("Carregamento na memória", f"{meta['load_duration_s']} s" if meta.get('load_duration_s') else "—")
                    dur_cols[2].metric("Leitura do prompt", f"{meta['prompt_eval_duration_s']} s" if meta.get('prompt_eval_duration_s') else "—")
                    dur_cols[3].metric("Geração da resposta", f"{meta['eval_duration_s']} s" if meta.get('eval_duration_s') else "—")
                    _ollama_info = []
                    if meta.get("done_reason"):
                        _ollama_info.append(f"Motivo de parada: `{meta['done_reason']}`")
                    if meta.get("model"):
                        _ollama_info.append(f"Modelo: `{meta['model']}`")
                    if meta.get("created_at"):
                        _ollama_info.append(f"Gerado em: `{meta['created_at']}`")
                    if _ollama_info:
                        st.caption("  ·  ".join(_ollama_info))

                else:  # Groq
                    # ── Tempos Groq ───────────────────────────────────────
                    # queue_time = tempo na fila dos servidores Groq antes de processar
                    st.caption(
                        "☁️ **Tempos de API cloud** — sem acesso ao hardware, medem o ciclo completo na infraestrutura Groq: "
                        "fila de espera, processamento do prompt e geração da resposta."
                    )
                    gtime_cols = st.columns(4)
                    gtime_cols[0].metric("Tempo na fila", f"{meta['queue_time_s']} s" if meta.get('queue_time_s') else "—",
                        help="Tempo que a requisição esperou na fila dos servidores Groq antes de começar a ser processada.")
                    gtime_cols[1].metric("Tempo total", f"{meta['total_time_s']} s" if meta.get('total_time_s') else "—")
                    gtime_cols[2].metric("Tempo do prompt", f"{meta['prompt_time_s']} s" if meta.get('prompt_time_s') else "—")
                    gtime_cols[3].metric("Tempo de geração", f"{meta['completion_time_s']} s" if meta.get('completion_time_s') else "—")

                    _groq_info = []
                    if meta.get("finish_reason"):
                        _groq_info.append(f"Finish reason: `{meta['finish_reason']}`")
                    if meta.get("completion_id"):
                        _groq_info.append(f"Completion ID: `{meta['completion_id']}`")
                    if meta.get("request_id"):
                        _groq_info.append(f"Request ID: `{meta['request_id']}`")
                    if meta.get("created"):
                        import datetime as _dt
                        _ts = _dt.datetime.fromtimestamp(meta["created"]).strftime("%d/%m/%Y %H:%M:%S")
                        _groq_info.append(f"Criado em: {_ts}")
                    if _groq_info:
                        st.caption("  ·  ".join(_groq_info))

# ──────────────────────────────────────────────
# Tab 2 — Dataset
# ──────────────────────────────────────────────
with tab_dataset:
    st.subheader("Classificação de Dataset")

    col_up, col_hint = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload do Dataset",
            type=["csv", "xlsx", "xls"],
            help="Suporta CSV (UTF-8) e Excel (.xlsx / .xls).",
            accept_multiple_files=False,
        )
    with col_hint:
        st.markdown("")
        st.markdown("")
        st.info("Após o upload, escolha abaixo qual **coluna** contém os textos a classificar.", icon="👇")

    if uploaded_file is not None:
        try:
            df = load_dataframe(uploaded_file)

            st.success(
                f"✅ Arquivo carregado: **{len(df):,} linhas** × **{len(df.columns)} colunas**"
            )

            # ── Seleção de coluna ─────────────────────────────────────────
            st.markdown("### 📌 Selecione a coluna com os textos")
            col_sel, col_count = st.columns([3, 1])
            with col_sel:
                column_name = st.selectbox(
                    "Coluna de texto para classificar",
                    options=df.columns.tolist(),
                    help=(
                        "Escolha qual coluna do dataset contém os textos que "
                        "serão enviados ao modelo para classificação."
                    ),
                )
            with col_count:
                valid_count = int(df[column_name].notna().sum())
                st.metric("Textos válidos", f"{valid_count:,}")

            with st.expander(f"👁️ Amostra da coluna **{column_name}** (5 primeiros)"):
                preview_df = df[[column_name]].head(5).copy()
                preview_df.index = range(1, len(preview_df) + 1)
                st.dataframe(preview_df, use_container_width=True)

            with st.expander("Ver todas as colunas do dataset (5 primeiras linhas)"):
                st.dataframe(df.head(5), use_container_width=True)

            st.divider()

            col_run2, col_clr2 = st.columns([2, 1])
            with col_run2:
                run_dataset = st.button(
                    "🚀 Classificar Dataset",
                    type="primary",
                    disabled=not st.session_state.api_ok or valid_count == 0,
                    use_container_width=True,
                )
            with col_clr2:
                if st.button("🗑️ Limpar resultados", use_container_width=True, key="clr_dataset"):
                    st.session_state.dataset_results = None

            if run_dataset:
                if not prompt_template.strip():
                    st.warning("Defina um prompt na barra lateral.")
                elif provider == "Groq" and not groq_key.strip():
                    st.warning("Insira a chave de API Groq na barra lateral.")
                else:
                    texts = df[column_name].dropna().astype(str).tolist()
                    total_texts = len(texts)

                    prog_bar = st.progress(0, text="Aguardando…")
                    stats_area = st.empty()

                    rows: list[dict] = []
                    error_count = 0

                    for i, row_text in enumerate(texts):
                        try:
                            if provider == "Groq":
                                row_cls = call_classify_text_groq(
                                    api_url, row_text, prompt_template, model_name,
                                    groq_key, temperature, top_p, int(max_tokens),
                                )
                            else:
                                row_cls = call_classify_text(
                                    api_url, row_text, prompt_template, model_name,
                                    temperature, top_p, int(top_k), int(max_tokens), repeat_penalty,
                                )
                            rows.append({"#": i + 1, "Texto": row_text, "Classificação": row_cls})
                        except requests.exceptions.HTTPError as http_exc:
                            error_count += 1
                            detail = http_exc.response.json().get("detail", str(http_exc))
                            rows.append({"#": i + 1, "Texto": row_text, "Classificação": f"ERRO: {detail}"})
                        except requests.exceptions.RequestException as req_exc:
                            error_count += 1
                            rows.append({"#": i + 1, "Texto": row_text, "Classificação": f"ERRO: {req_exc}"})

                        pct_done = (i + 1) / total_texts
                        prog_bar.progress(
                            pct_done,
                            text=f"Processando **{i + 1}/{total_texts}** — {pct_done:.0%}",
                        )
                        stats_area.caption(
                            f"✅ Concluídos: {i + 1 - error_count}  |  ❌ Erros: {error_count}"
                        )

                    prog_bar.progress(1.0, text="Concluído!")
                    st.session_state.dataset_results = rows

            if st.session_state.dataset_results:
                result_df = pd.DataFrame(st.session_state.dataset_results)
                total_r = len(result_df)
                errors_r = int(result_df["Classificação"].str.startswith("ERRO:").sum())

                st.success(
                    f"✅ **{total_r - errors_r}/{total_r}** classificados com sucesso"
                    + (f" | ❌ {errors_r} erro(s)" if errors_r > 0 else "")
                )

                non_error = result_df[~result_df["Classificação"].str.startswith("ERRO:").astype(bool)]
                if not non_error.empty:
                    with st.expander("📊 Distribuição das classificações"):
                        counts = non_error["Classificação"].value_counts().reset_index()
                        counts.columns = ["Classificação", "Quantidade"]
                        col_chart, col_tbl = st.columns([2, 1])
                        with col_chart:
                            st.bar_chart(counts.set_index("Classificação"))
                        with col_tbl:
                            st.dataframe(counts, use_container_width=True)

                st.dataframe(result_df, use_container_width=True, height=350)

                st.download_button(
                    label="📥 Download Resultados (CSV)",
                    data=to_csv_bytes(result_df),
                    file_name="resultados_classificacao.csv",
                    mime="text/csv",
                )

        except ValueError as load_exc:
            st.error(f"❌ Erro ao carregar o arquivo: {load_exc}")
    else:
        st.info(
            "📂 Faça upload de um arquivo CSV ou Excel para classificação em lote.",
            icon="ℹ️",
        )
