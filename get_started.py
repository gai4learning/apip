"""Quick-reference Streamlit page generated from the shared model catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from config.model_catalog import (
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_IMAGE_MODELS,
    MODEL_CATALOG,
    REGION_NAMES,
    Region,
    get_model,
)

if TYPE_CHECKING:
    from config.settings import Settings


def show_get_started(settings: Settings) -> None:
    st.title("Get Started")
    st.error("Local-only learning tool: keep it inside the Codespace and do not expose public ingress.")
    st.write(
        "**Starter** is a limited sampler for learning, smoke tests, and small non-sensitive model "
        "comparisons. It is not for production, sustained applications, repository-scale agents, or "
        "synchronized class activity."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Call rate", "10 / 60 sec")
    col2.metric("Call quota", "100 / 7 days")
    col3.metric("LLM token rate", "250,000 TPM*")
    col4.metric("Monthly LLM tokens", "5,000,000*")
    st.caption("*Where the configured operation supports the CUHK language-model token policy; counters are per APIM subscription.")

    st.subheader("Regional endpoints")
    st.code(f"EUS2  {settings.eus2_base_url}\nWUS3  {settings.wus3_base_url}")

    st.subheader("Recommended first tests")
    recommended_models = (
        get_model(Region.EUS2, DEFAULT_CHAT_MODELS[Region.EUS2]),
        get_model(Region.EUS2, DEFAULT_IMAGE_MODELS[Region.EUS2]),
        get_model(Region.WUS3, DEFAULT_IMAGE_MODELS[Region.WUS3]),
        get_model(Region.EUS2, DEFAULT_EMBEDDING_MODEL),
    )
    recommendations = [
        {
            "Capability": model.capability.value,
            "Region": model.region.value,
            "Model": model.model_id,
            "Operation": model.operation.value,
        }
        for model in recommended_models
    ]
    st.dataframe(recommendations, hide_index=True)
    st.info("Different capabilities require different operations. Never send specialized models to Chat Completions.")

    for region in Region:
        with st.expander(f"{REGION_NAMES[region]} model summary"):
            rows = [
                {
                    "Model": model.model_id,
                    "Capability": model.capability.value,
                    "Operation": model.operation.value,
                    "UI status": model.ui_status,
                    "Validation": model.validation_status.value,
                }
                for model in MODEL_CATALOG
                if model.region is region
            ]
            st.dataframe(rows, hide_index=True)

    st.subheader("Key security")
    st.warning(
        "Use a CUHK APIM subscription key—not a Foundry backend key. Keep `.env` out of Git, use the "
        "password-masked field only as a temporary override, and rotate the key if exposure is suspected."
    )
    st.subheader("When Starter is not enough")
    st.write(
        "Do not share one Starter subscription across a class. Teachers, TAs, FYP leaders, research "
        "projects, and sustained applications should prepare the request details on the **Special "
        "Access** page and submit them through the official ITSC service channel."
    )
