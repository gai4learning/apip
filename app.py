"""Local-only Streamlit learning tool for CUHK Foundry Models EUS2/WUS3."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from clients.cuhk_apim_client import CUHKAPIMClient
from config.model_catalog import (
    DEFAULT_EMBEDDING_MODEL,
    MODEL_CATALOG,
    MODEL_ROUTER_MODEL_ID,
    REGION_NAMES,
    Capability,
    Operation,
    Region,
    chat_models,
    embedding_models,
    image_models,
)
from config.settings import Settings, load_settings
from services.chat_service import ChatResult, ChatService
from services.embedding_service import (
    EmbeddingResult,
    EmbeddingService,
    similarity_matrix,
)
from services.image_service import ImageResult, ImageService
from utils.errors import APIMError, sanitized_unexpected_error
from utils.security import configure_local_logging, mask_api_key
from utils.usage import HeaderSummary, safe_session_export

SMOKE_TEST_PROMPT = "Reply exactly with: CUHK APIM test successful."
IMAGE_TEST_PROMPT = (
    "A clean abstract illustration of a university digital learning platform, "
    "blue and purple palette, simple geometric forms, no text"
)
PAGES = (
    "Get Started",
    "Chat",
    "Model Router",
    "Image Generation",
    "Embeddings",
    "Usage and Limits",
    "Special Access",
    "About / Safety",
)


def _region_select(label: str, default: Region, key: str) -> Region:
    regions = list(Region)
    return st.selectbox(
        label,
        regions,
        index=regions.index(default),
        format_func=lambda region: REGION_NAMES[region],
        key=key,
    )


def _client(settings: Settings, region: Region, api_key: str, logger: Any) -> CUHKAPIMClient:
    return CUHKAPIMClient(settings.base_url(region), api_key, logger=logger)


def _key_input(settings: Settings) -> str:
    st.sidebar.subheader("Local credential")
    entered = st.sidebar.text_input(
        "APIM subscription key (optional override)",
        type="password",
        help="Used only for requests in this Streamlit session. It is not logged or exported.",
    )
    key = entered.strip() or settings.api_key
    st.sidebar.caption(mask_api_key(key))
    st.sidebar.caption("Use an APIM subscription key—not a Foundry backend key.")
    return key


def _render_status(model: Any) -> None:
    st.caption(
        f"{model.ui_status} · {model.validation_status.value} · "
        f"{model.api_name} · {model.operation.value}"
    )


def _render_headers(headers: HeaderSummary) -> None:
    st.subheader("Response-header summary")
    if headers.routing:
        st.write("**Routing and request identifiers**")
        st.json(headers.routing)
    else:
        st.caption("No supported routing or request-ID headers were returned.")
    if headers.cuhk_allowance:
        st.write("**CUHK APIM allowance**")
        st.json(headers.cuhk_allowance)
    else:
        st.caption("No CUHK APIM allowance headers were returned for this operation.")
    if headers.backend_capacity:
        st.write("**Foundry backend capacity**")
        st.json(headers.backend_capacity)
    else:
        st.caption("No backend x-ratelimit headers were returned.")


def _render_usage(usage: dict[str, int]) -> None:
    st.subheader("Response-body usage")
    if not usage:
        st.caption("This operation did not return a recognized usage object.")
        return
    labels = {
        "prompt_tokens": "Prompt/input tokens",
        "completion_tokens": "Completion/output tokens",
        "reasoning_tokens": "Reasoning tokens",
        "total_tokens": "Total tokens",
    }
    columns = st.columns(len(usage))
    for column, (name, value) in zip(columns, usage.items()):
        column.metric(labels.get(name, name), value)


def _render_api_error(error: APIMError, region: Region, operation: Operation) -> None:
    st.error(str(error))
    if error.code == "unknown_model" or (error.status_code == 400 and "model" in error.code.lower()):
        valid = [model.model_id for model in MODEL_CATALOG if model.region is region and model.operation is operation]
        st.info("Valid regional choices: " + ", ".join(valid))
    if error.status_code == 404 and operation not in {Operation.CHAT_COMPLETIONS}:
        st.info(
            "The CUHK APIM contract may not expose this specialized operation yet. "
            "The app will not reroute it through Chat Completions."
        )
    if error.headers:
        _render_headers(error.headers)


def _record(stats: dict[str, Any]) -> None:
    records = st.session_state.setdefault("usage_records", [])
    safe = dict(stats)
    safe["timestamp_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    records.append(safe)


def _chat_record(result: ChatResult) -> None:
    _record(
        {
            "operation": Operation.CHAT_COMPLETIONS.value,
            "region": result.region.value,
            "requested_model": result.requested_model,
            "served_model": result.served_model,
            "status_code": result.status_code,
            "finish_reason": result.finish_reason,
            "latency_ms": result.latency_ms,
            "request_id": result.headers.request_id,
            "apim_request_id": result.headers.apim_request_id,
            **result.usage,
        }
    )


def _render_chat_result(result: ChatResult) -> None:
    st.subheader("Sanitized response")
    if result.text:
        st.write(result.text)
    if result.empty_output_guidance:
        st.warning(result.empty_output_guidance)
    columns = st.columns(5)
    columns[0].metric("HTTP status", result.status_code)
    columns[1].metric("Requested model", result.requested_model)
    columns[2].metric("Served model/version", result.served_model or "Not returned")
    columns[3].metric("Region", REGION_NAMES[result.region])
    columns[4].metric("Latency", f"{result.latency_ms} ms")
    st.write(f"**Finish reason:** {result.finish_reason or 'Not returned'}")
    st.write(
        "**Responsible-AI invocation:** "
        + result.headers.routing.get("x-ms-rai-invoked", "Not returned")
    )
    _render_usage(result.usage)
    _render_headers(result.headers)


def show_chat(settings: Settings, api_key: str, logger: Any) -> None:
    st.title("Chat Completions")
    st.caption("Capability-aware regional testing through CUHK APIM OpenAI v1.")
    region = _region_select("Region", settings.default_region, "chat_region")
    choices = chat_models(region)
    ids = [model.model_id for model in choices]
    default_id = settings.default_chat_model if settings.default_chat_model in ids else ids[0]
    selected_id = st.selectbox(
        "Model",
        ids,
        index=ids.index(default_id),
        format_func=lambda model_id: next(model.display_name for model in choices if model.model_id == model_id),
    )
    model = next(model for model in choices if model.model_id == selected_id)
    _render_status(model)
    if st.button("Clear conversation"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_chat_result", None)
        st.rerun()

    with st.form("chat_form"):
        system_instruction = st.text_area(
            "System instruction", "You are a concise, responsible learning assistant.", max_chars=4_000
        )
        user_prompt = st.text_area("User prompt", SMOKE_TEST_PROMPT, max_chars=12_000)
        max_completion_tokens = st.number_input(
            "Maximum completion tokens", min_value=1, max_value=16_384, value=100, step=25
        )
        temperature = None
        if model.supports_temperature:
            temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
        submitted = st.form_submit_button("Send chat request", type="primary")
    if submitted:
        try:
            result = ChatService(_client(settings, region, api_key, logger)).complete(
                region=region,
                model_id=selected_id,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                max_completion_tokens=int(max_completion_tokens),
                temperature=temperature,
                conversation=st.session_state.get("chat_history", []),
            )
            st.session_state.setdefault("chat_history", []).extend(
                [
                    {"role": "user", "content": user_prompt.strip()},
                    {"role": "assistant", "content": result.text},
                ]
            )
            st.session_state["last_chat_result"] = result
            _chat_record(result)
        except APIMError as error:
            _render_api_error(error, region, Operation.CHAT_COMPLETIONS)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001 - final sanitized UI boundary
            logger.error("Unexpected chat failure; request and response bodies omitted")
            st.error(sanitized_unexpected_error(error, api_key))
    result = st.session_state.get("last_chat_result")
    if isinstance(result, ChatResult):
        _render_chat_result(result)
    history = st.session_state.get("chat_history", [])
    if history:
        with st.expander("Conversation in this local session"):
            for item in history:
                st.write(f"**{item['role'].title()}:**")
                st.write(item["content"])


def show_model_router(settings: Settings, api_key: str, logger: Any) -> None:
    st.title("Model Router")
    st.info(
        f"The requested deployment is **{MODEL_ROUTER_MODEL_ID}**. The router may select another underlying model, "
        "which can be reported in response headers when supplied. Reasoning tokens can consume the "
        "completion allowance; a low maximum may yield empty visible output with `finish_reason: length`."
    )
    st.warning(f"Do not calculate cost from the {MODEL_ROUTER_MODEL_ID} alias alone.")
    model = next(model for model in MODEL_CATALOG if model.model_id == MODEL_ROUTER_MODEL_ID)
    _render_status(model)
    with st.form("router_form"):
        system_instruction = st.text_area(
            "System instruction", "Answer clearly and briefly.", max_chars=4_000, key="router_system"
        )
        user_prompt = st.text_area(
            "User prompt",
            "Reply in exactly one short sentence: Why is an API gateway useful?",
            max_chars=12_000,
            key="router_prompt",
        )
        allowance = st.number_input(
            "Maximum completion tokens", min_value=100, max_value=16_384, value=1_000, step=100
        )
        submitted = st.form_submit_button("Test Model Router", type="primary")
    if submitted:
        try:
            result = ChatService(_client(settings, Region.EUS2, api_key, logger)).complete(
                region=Region.EUS2,
                model_id=MODEL_ROUTER_MODEL_ID,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                max_completion_tokens=int(allowance),
            )
            st.session_state["last_router_result"] = result
            _chat_record(result)
        except APIMError as error:
            _render_api_error(error, Region.EUS2, Operation.CHAT_COMPLETIONS)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001 - final sanitized UI boundary
            logger.error("Unexpected Model Router failure; bodies omitted")
            st.error(sanitized_unexpected_error(error, api_key))
    result = st.session_state.get("last_router_result")
    if isinstance(result, ChatResult):
        _render_chat_result(result)
        routing = {
            key: value
            for key, value in result.headers.routing.items()
            if key.startswith("x-model-router-")
        }
        if routing:
            st.subheader("Router decision headers")
            st.json(routing)


def show_images(settings: Settings, api_key: str, logger: Any) -> None:
    st.title("Image Generation")
    st.caption("Images are decoded in memory, validated, displayed, and never written to logs.")
    region = _region_select("Region", settings.default_image_region, "image_region")
    choices = image_models(region)
    ids = [model.model_id for model in choices]
    default_id = settings.default_image_model if settings.default_image_model in ids else ids[0]
    selected_id = st.selectbox("Image model", ids, index=ids.index(default_id))
    model = choices[ids.index(selected_id)]
    _render_status(model)
    st.caption(f"Endpoint: {settings.base_url(region)}/{Operation.IMAGE_GENERATION.value}")
    with st.form("image_form"):
        prompt = st.text_area("Prompt", IMAGE_TEST_PROMPT, max_chars=4_000)
        size = st.selectbox("Size", ("1024x1024", "1536x1024", "1024x1536"))
        quality = st.selectbox("Quality", ("low", "medium", "high"))
        output_format = st.selectbox("Output format", ("png", "jpeg", "webp"))
        starter_mode = st.checkbox("Starter mode", value=True, help="Constrains n to 1.")
        count = st.number_input(
            "Image count", min_value=1, max_value=1 if starter_mode else 4, value=1, step=1
        )
        submitted = st.form_submit_button("Generate image", type="primary")
    if submitted:
        try:
            result = ImageService(_client(settings, region, api_key, logger)).generate(
                region=region,
                model_id=selected_id,
                prompt=prompt,
                size=size,
                quality=quality,
                output_format=output_format,
                n=int(count),
                starter_mode=starter_mode,
            )
            st.session_state["last_image_result"] = result
            _record(
                {
                    "operation": Operation.IMAGE_GENERATION.value,
                    "region": result.region.value,
                    "requested_model": result.requested_model,
                    "status_code": result.status_code,
                    "latency_ms": result.latency_ms,
                    "image_size": result.size,
                    "quality": result.quality,
                    "output_format": result.output_format,
                    "count": len(result.images),
                    "request_id": result.headers.request_id,
                    "apim_request_id": result.headers.apim_request_id,
                    **result.usage,
                }
            )
        except APIMError as error:
            _render_api_error(error, region, Operation.IMAGE_GENERATION)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001 - final sanitized UI boundary
            logger.error("Unexpected image failure; prompt and image content omitted")
            st.error(sanitized_unexpected_error(error, api_key))
    result = st.session_state.get("last_image_result")
    if isinstance(result, ImageResult):
        columns = st.columns(8)
        values = (
            ("HTTP", result.status_code), ("Model", result.requested_model),
            ("Region", result.region.value), ("Size", result.size),
            ("Quality", result.quality), ("Format", result.output_format),
            ("Count", len(result.images)), ("Latency", f"{result.latency_ms} ms"),
        )
        for column, (label, value) in zip(columns, values):
            column.metric(label, value)
        for index, image in enumerate(result.images, start=1):
            try:
                st.image(image.content, caption=f"Generated image {index}")
            except Exception:  # noqa: BLE001 - Streamlit renderer boundary
                st.error("The validated image could not be rendered locally.")
            st.download_button(
                f"Download image {index}",
                data=image.content,
                file_name=f"cuhk-image-{index}.{image.extension}",
                mime=image.media_type,
            )
        _render_usage(result.usage)
        _render_headers(result.headers)
    st.info(
        "Image cost is affected by image-specific settings such as model, quality, size, and count. "
        "Do not estimate image cost from language-token headers."
    )


def show_embeddings(settings: Settings, api_key: str, logger: Any) -> None:
    st.title("Embeddings (East US 2)")
    st.caption("Embeddings use the EUS2 `/embeddings` operation and are never routed through chat.")
    choices = embedding_models()
    ids = [model.model_id for model in choices]
    default = settings.default_embedding_model if settings.default_embedding_model in ids else ids[0]
    selected_id = st.selectbox("Embedding model", ids, index=ids.index(default))
    model = choices[ids.index(selected_id)]
    _render_status(model)
    if model.legacy_compatibility:
        st.warning(
            f"Legacy compatibility only. Evaluate {DEFAULT_EMBEDDING_MODEL} first for new demonstrations."
        )
    mode = st.radio("Mode", ("Inspect one embedding", "Compare multiple texts"), horizontal=True)
    if mode == "Inspect one embedding":
        text = st.text_area("Input text", "CUHK AI API Portal embedding test", max_chars=4_000)
        submitted = st.button("Create embedding", type="primary")
        texts = [text]
    else:
        count = st.number_input("Number of texts", min_value=2, max_value=8, value=3, step=1)
        examples = (
            "Digital learning infrastructure",
            "Generative AI for university teaching",
            "Secure API governance",
        )
        texts = [
            st.text_area(
                f"Text {index + 1}",
                examples[index] if index < len(examples) else "",
                max_chars=4_000,
                key=f"embedding_text_{index}",
            )
            for index in range(int(count))
        ]
        submitted = st.button("Compare embeddings", type="primary")
        st.caption(
            "Similarity scores are mathematical comparisons—not factual or quality judgments—and must "
            "not be interpreted as plagiarism, authorship, intent, or academic misconduct."
        )
    if submitted:
        try:
            result = EmbeddingService(_client(settings, Region.EUS2, api_key, logger)).embed(
                model_id=selected_id,
                texts=texts,
            )
            st.session_state["last_embedding_result"] = result
            st.session_state["last_embedding_mode"] = mode
            _record(
                {
                    "operation": Operation.EMBEDDINGS.value,
                    "region": Region.EUS2.value,
                    "requested_model": result.requested_model,
                    "served_model": result.response_model,
                    "status_code": result.status_code,
                    "latency_ms": result.latency_ms,
                    "vector_count": result.vector_count,
                    "vector_dimension": result.dimension,
                    "request_id": result.headers.request_id,
                    "apim_request_id": result.headers.apim_request_id,
                    **result.usage,
                }
            )
        except APIMError as error:
            _render_api_error(error, Region.EUS2, Operation.EMBEDDINGS)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001 - final sanitized UI boundary
            logger.error("Unexpected embedding failure; inputs and vectors omitted")
            st.error(sanitized_unexpected_error(error, api_key))
    result = st.session_state.get("last_embedding_result")
    result_mode = st.session_state.get("last_embedding_mode")
    if isinstance(result, EmbeddingResult):
        columns = st.columns(6)
        metrics = (
            ("Model", result.response_model or result.requested_model),
            ("Object", result.object_type or "Not returned"),
            ("Vectors", result.vector_count),
            ("Dimension", result.dimension),
            ("HTTP", result.status_code),
            ("Latency", f"{result.latency_ms} ms"),
        )
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, value)
        if result_mode == "Compare multiple texts":
            try:
                matrix = similarity_matrix(result.vectors)
                labels = [f"Text {index + 1}" for index in range(result.vector_count)]
                table = [
                    {"Text": labels[row], **{
                        labels[column]: f"{matrix[row][column]:.4f}"
                        for column in range(len(labels))
                    }}
                    for row in range(len(labels))
                ]
                st.dataframe(table, hide_index=True)
            except ValueError as error:
                st.error(str(error))
        if st.checkbox("Show vector sample (first and last 8 values)"):
            sample = result.sample()
            st.json({key: list(values) for key, values in sample.items()})
        _render_usage(result.usage)
        _render_headers(result.headers)


def show_usage_limits() -> None:
    st.title("Usage and Limits")
    st.warning("These are product-policy limits, not application-side counters or a billing ledger.")
    limits = [
        {"Control": "Starter call rate", "Current presentation": "10 calls per 60 seconds", "Scope": "Per APIM subscription", "When reached": "HTTP 429"},
        {"Control": "Starter call quota", "Current presentation": "100 calls per seven days", "Scope": "Per APIM subscription", "When reached": "HTTP 403"},
        {"Control": "Language-model token rate", "Current presentation": "250,000 TPM where configured", "Scope": "Per APIM subscription", "When reached": "HTTP 429"},
        {"Control": "Language-model monthly quota", "Current presentation": "5,000,000 tokens", "Scope": "Per APIM subscription", "When reached": "HTTP 403"},
        {"Control": "Backend capacity", "Current presentation": "Deployment-specific", "Scope": "Foundry backend", "When reached": "Backend 429"},
    ]
    st.dataframe(limits, hide_index=True)
    st.write(
        "Product call limits and API token limits are separate; the first limit reached stops usage. "
        "One VS Code Agent task can make multiple API requests. Image generation should initially be "
        "assessed through request-rate and call quota controls, not language-token pricing. Embeddings "
        "may not emit the same metrics as Chat Completions."
    )
    records = st.session_state.get("usage_records", [])
    st.subheader("Application-side session statistics")
    st.caption("Local session metadata only; it is distinct from CUHK APIM allowance and backend capacity.")
    export = safe_session_export(records)
    if records:
        st.dataframe(export["records"], hide_index=True)
    else:
        st.info("No requests have been completed in this local session.")
    st.download_button(
        "Download sanitized session statistics",
        data=json.dumps(export, indent=2),
        file_name="cuhk-apip-session-statistics.json",
        mime="application/json",
    )


def show_special_access() -> None:
    st.title("Special Access")
    st.warning(
        "A shared Starter subscription must not be used for a synchronized class. Teachers, TAs, FYP "
        "leaders, research projects, and sustained applications should request a special product."
    )
    st.write(
        "No authoritative mailbox or request form is documented in this repository. Use the official "
        "ITSC service channel and adapt this guidance template."
    )
    template = """Request title: <course/project/application name>
Requestor and unit: <name and unit>
Accountable teacher, supervisor, PI, or service owner: <name/role>
Course or project: <identifier and description>
Purpose: <use case and why Starter is insufficient>
Number and type of users: <teachers/TAs/students/project members/applications>
Requested start and end dates: <dates>
Required models and operations: <chat/Responses/image/embeddings/audio/realtime/etc.>
Expected simultaneous users: <estimate>
Expected calls per minute: <estimate>
Expected monthly usage or budget ceiling: <estimate>
Data classification: <classification and required review>
Client type: <portal/VS Code/SDK/local app/service>
Required reporting: <dimensions and frequency>
Funding or cost ownership: <owner/project code if applicable>
Approvals: <teacher/supervisor/PI/security/privacy/procurement/etc.>
Operational contact: <primary and backup contact>"""
    st.code(template)


def show_about_safety() -> None:
    st.title("About / Safety")
    st.error("Local-only learning and testing tool. Do not expose it through public ingress or host it as a service.")
    st.markdown(
        """
- Use only the APIM subscription key assigned to your product or project. Never request a Foundry backend key.
- Never put keys in source control, screenshots, chat, notebooks, exported statistics, or documentation.
- If key exposure is suspected, stop using it and request rotation; do not display it for diagnosis.
- Use synthetic or de-identified data for initial tests. Do not submit restricted data without approval.
- Model output can be inaccurate and must be reviewed. Follow CUHK responsible-use and academic-honesty requirements.
- Similarity scores do not establish plagiarism, authorship, intent, or misconduct.
- Agent tools may read files and make multiple calls. Scope permissions narrowly and inspect each request.
"""
    )
    with st.expander("Advanced / operation-specific models"):
        rows = [
            {
                "Model": model.model_id,
                "Region": model.region.value,
                "Capability": model.capability.value,
                "Required operation": model.operation.value,
                "UI support": model.ui_status,
                "Validation": model.validation_status.value,
            }
            for model in MODEL_CATALOG
            if model.operation not in {
                Operation.CHAT_COMPLETIONS,
                Operation.IMAGE_GENERATION,
                Operation.EMBEDDINGS,
            }
            or model.capability in {Capability.REALTIME, Capability.TRANSLATION, Capability.TRANSCRIPTION}
        ]
        st.dataframe(rows, hide_index=True)
        st.caption(
            "Realtime, translation, whisper, transcription, and coding/Responses deployments are not "
            "routed through Chat Completions. Their operation-specific UIs are not yet implemented."
        )


def main() -> None:
    global st
    import streamlit as st
    from dotenv import load_dotenv

    from get_started import show_get_started

    load_dotenv()
    st.set_page_config(page_title="CUHK Foundry API Learning Tool", layout="wide")
    st.error("LOCAL-ONLY: run inside your Codespace. Do not publish or expose this Streamlit app.")
    try:
        settings = load_settings()
    except ValueError as error:
        st.error(f"Configuration error: {error}")
        st.stop()
    logger = configure_local_logging()
    api_key = _key_input(settings)
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.caption("EUS2 and WUS3 requests are never automatically failed over between regions.")

    if page == "Get Started":
        show_get_started(settings)
    elif page == "Chat":
        show_chat(settings, api_key, logger)
    elif page == "Model Router":
        show_model_router(settings, api_key, logger)
    elif page == "Image Generation":
        show_images(settings, api_key, logger)
    elif page == "Embeddings":
        show_embeddings(settings, api_key, logger)
    elif page == "Usage and Limits":
        show_usage_limits()
    elif page == "Special Access":
        show_special_access()
    else:
        show_about_safety()


if __name__ == "__main__":
    main()
