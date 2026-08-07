# CUHK Foundry API Local Learning Tool

A local-only Streamlit application for learning, smoke-testing, and comparing the current CUHK Foundry Models APIs through CUHK API Management (APIM). It demonstrates capability-aware regional routing for Chat Completions, Model Router, image generation, and embeddings without exposing credentials or sending specialized models to the wrong operation.

> **Migration note for 2026–27 classes and new users:** use **CUHK Foundry Models EUS2** or **CUHK Foundry Models WUS3** through the OpenAI v1 routes documented here. Do not start new demonstrations with retired legacy Azure OpenAI deployment URLs, dated `api-version` parameters, or superseded model lists.

> **Local-only warning:** run this app only inside your Codespace for individual learning and testing. Do not add public ingress, publish it as a hosted application, or use it as a production service.

## 1. Project purpose

The tool provides:

- region- and capability-aware model selection from one typed catalogue;
- Chat Completions smoke tests for EUS2 and WUS3;
- a dedicated Model Router explanation and test;
- regional image-generation tests with safe in-memory Base64 decoding;
- EUS2 embedding inspection and bounded multi-text cosine comparison;
- separate views for response-body usage, CUHK APIM allowance headers, Foundry backend capacity headers, and local session statistics;
- quick-start, limits, special-access, security, responsible-use, and academic-honesty guidance.

Starter is a limited sampler. It is not intended for production, sustained repository review, synchronized teaching, or shared credentials.

## 2. Security notice

Use only the **APIM subscription key** issued for the relevant product or project. Do not request, distribute, or configure a Foundry backend key.

- Never commit `.env`, keys, bearer tokens, or local secret files.
- Never share a key in email, chat, screenshots, notebooks, source code, logs, or documentation.
- The optional UI key field is password-masked and is not included in app exports or logs.
- Logs contain operation metadata only. They exclude prompts, complete request/response bodies, headers, keys, Base64 images, embedding vectors, and source files.
- Exported session statistics contain allowlisted metadata only.
- Use synthetic or de-identified inputs for initial testing. Do not submit confidential, personal, assessment-restricted, research-sensitive, or regulated data without an approved product and handling conditions.
- If exposure is suspected, stop using the key and request rotation. Do not display the key while diagnosing the problem.
- Model output is rendered as plain text by default, so untrusted responses cannot create clickable Markdown links or remote Markdown resources.
- Chat history and local usage records are bounded in memory; generated images and statistics can be cleared from the session.
- Resolver-generated dependency locks pin exact versions and artifact hashes in `requirements.lock` and `requirements-dev.lock`; CI regenerates, clean-installs, compatibility-checks, and vulnerability-audits them.

An APIM subscription credential is not an Azure subscription and is not a Foundry backend credential.

## 3. Environment-variable setup

Copy the template and fill only your local `.env`:

```bash
cp .env.example .env
```

```dotenv
# APIM subscription key—not a Foundry backend key
CUHK_APIM_API_KEY=
CUHK_EUS2_BASE_URL=https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1
CUHK_WUS3_BASE_URL=https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1
CUHK_DEFAULT_REGION=EUS2
CUHK_DEFAULT_CHAT_MODEL=gpt-5.4-mini
CUHK_DEFAULT_IMAGE_MODEL=gpt-image-2
CUHK_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

The key may instead be entered as a temporary password-masked UI override. Only `CUHK_APIM_API_KEY` is accepted; the generic `AZURE_API_KEY` name is deliberately ignored so an unrelated Azure or backend credential cannot be sent to APIM. Do not put the key in browser-local storage or source control. The application validates that a key is nonblank only when a request is made, so the UI and tests start without a live key.

## 4. Installation

Python **3.11.15** is required. Codespaces may start with a different Python version and may not include `pyenv`. Check the default interpreter first:

```bash
python3 --version
```

If it is not Python 3.11.15, install the pinned `uv` tool with the available interpreter and use it to provision the required version:

```bash
python3 -m pip install --user "uv==0.12.1"
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11.15
uv venv --python 3.11.15 .venv
source .venv/bin/activate
python --version  # Must report Python 3.11.15
uv pip install --require-hashes -r requirements.lock
```

To include the locked development and test tools, install the development graph instead of the runtime graph:

```bash
uv pip install --require-hashes -r requirements-dev.lock
```

The repository's `.python-version` records the required version for compatible version managers, but it does not install Python or guarantee that `pyenv` is available.

The checked-in files are resolver-generated, hash-locked dependency graphs for Python 3.11 on Linux Codespaces. `requirements.in` and `requirements-dev.in` record direct intent; `requirements.txt` remains a backward-compatible development entry point. The local checker validates exact direct versions and requires artifact hashes for every locked package:

```bash
python scripts/check_dependency_lock.py
```

In a connected, trusted Codespace, regenerate both hash-locked files with the pinned resolver whenever direct dependencies change:

```bash
python -m pip install 'uv==0.12.1'
uv pip compile requirements.in \
  --python-version 3.11.15 \
  --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes \
  -o requirements.lock
uv pip compile requirements-dev.in \
  --python-version 3.11.15 \
  --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes \
  -o requirements-dev.lock
python scripts/check_dependency_lock.py
```

Audit the resulting runtime graph with the reviewed audit tool version:

```bash
uvx --from pip-audit==2.10.1 pip-audit -r requirements.lock
```

The GitHub security-verification workflow independently regenerates both lock files and compares them byte-for-byte with the committed versions. It then enforces hashes during a clean installation, runs `pip check`, executes the pinned audit, and runs lint, compilation, and the full mocked test suite.

Exact pins and hashes do not by themselves prove that dependencies remain vulnerability-free. Keep successful resolver comparison, clean installation, `python -m pip check`, and audit results as merge gates.

## 5. Run the Streamlit app

```bash
streamlit run app.py
```

Tracked Streamlit configuration binds the process to `127.0.0.1`, enables CORS and XSRF protection, disables static file serving and usage telemetry, and the app refuses a non-loopback server address. **The Codespaces forwarded port must still remain private** because a public forwarding proxy can reach a loopback-bound process. Do not add a public tunnel, public ingress, or hosted authentication layer to this learning tool.

Navigation:

- Get Started
- Chat
- Model Router
- Image Generation
- Embeddings
- Usage and Limits
- Special Access
- About / Safety

## 6. Current regional endpoints

| API | OpenAI-compatible base URL |
|---|---|
| CUHK Foundry Models EUS2 | `https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1` |
| CUHK Foundry Models WUS3 | `https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1` |

Required operation paths:

| Capability | Request |
|---|---|
| Chat Completions | `POST <regional-base>/chat/completions` |
| Responses | `POST <regional-base>/responses` |
| Image generation | `POST <regional-base>/images/generations` |
| Embeddings | `POST <EUS2-base>/embeddings` |

Requests send the subscription credential only as `api-key: <subscription key>`. Normal OpenAI v1 Chat Completions requests do not append a dated `api-version` parameter. The app does not automatically fail over between regions.

## 7. Regional model catalogue

The code source of truth is [`config/model_catalog.py`](config/model_catalog.py). “Validated” below means the model-operation path was externally validated as stated in the supplied 4 August 2026 guide. Regional mapping alone does not imply a live operation test.

### East US 2

| Deployment | Capability | Initial operation / UI handling | Status |
|---|---|---|---|
| `gpt-5.4-nano` | Language | Chat Completions; Responses support requires separate operation validation | Available for testing |
| `gpt-5.4-mini` | Language | Chat Completions | Validated |
| `gpt-5.4` | Language | Chat Completions | Validated |
| `gpt-5.4-pro` | Language | Chat Completions combination pending | Available for testing |
| `gpt-5.3-codex` | Coding | Responses; UI not yet implemented | Validation pending |
| `model-router` | Routing | Chat Completions | Validated |
| `gpt-image-2` | Image | Images Generations | Available for testing |
| `gpt-realtime-2` | Realtime/audio | Operation-specific realtime session; UI not implemented | Validation pending |
| `gpt-realtime-2.1` | Realtime/audio | Operation-specific realtime session; UI not implemented | Validation pending |
| `gpt-realtime-2.1-mini` | Realtime/audio | Operation-specific realtime session; UI not implemented | Validation pending |
| `gpt-realtime-translate` | Translation/audio | Matching realtime/audio operation; UI not implemented | Validation pending |
| `gpt-realtime-whisper` | Speech recognition | Matching realtime/audio operation; UI not implemented | Validation pending |
| `gpt-4o-transcribe` | Transcription | Multipart transcription operation; UI not implemented | Validation pending |
| `text-embedding-3-small` | Embeddings | Embeddings | Available for testing; recommended first |
| `text-embedding-3-large` | Embeddings | Embeddings | Available for testing |
| `text-embedding-ada-002` | Embeddings | Embeddings | Legacy compatibility; validation pending |

### West US 3

| Deployment | Capability | Initial operation | Status |
|---|---|---|---|
| `gpt-5.6-sol` | Chat | Chat Completions | Validated |
| `gpt-5.6-luna` | Chat | Chat Completions | Available for testing |
| `gpt-5.6-terra` | Chat | Chat Completions | Available for testing |
| `gpt-image-1.5` | Image | Images Generations | Available for testing |

Availability and served versions can change. Check the official API portal before creating a dependency. The app prevents invalid regional image and embedding combinations.

## 8. Capability-to-operation mapping

Do not reuse one request body for every deployment.

| Capability/workflow | Operation | Output/request field | App support |
|---|---|---|---|
| GPT-5.4/GPT-5.6 chat | Chat Completions | `max_completion_tokens` | Implemented |
| Model Router chat | Chat Completions | `max_completion_tokens` | Implemented |
| Responses workflows | Responses | `max_output_tokens` | Not yet implemented |
| Image generation | Images Generations | image-specific schema | Implemented |
| Embeddings | Embeddings | `input` string or array | Implemented for EUS2 |
| Transcription | Audio transcription | multipart file/audio schema | Not yet implemented |
| Realtime/translation | Matching realtime/audio operation | operation-specific protocol | Not yet implemented |

The app does not route realtime, translation, whisper, transcription, coding/Responses, image, or embedding deployments through Chat Completions and does not fabricate templates for unimplemented operations.

## 9. Chat test

Select **Chat**, **East US 2**, and `gpt-5.4-mini`. The default smoke test is:

```json
{
  "model": "gpt-5.4-mini",
  "messages": [
    {"role": "user", "content": "Reply exactly with: CUHK APIM test successful."}
  ],
  "max_completion_tokens": 100
}
```

For WUS3, select `gpt-5.6-sol`; the operation remains `/chat/completions`. Model output is rendered as plain text. The UI displays HTTP status, requested and served model/version, region, finish reason, token usage including reasoning tokens when present, request IDs, responsible-AI indication, and latency. It shows only bounded, allowlisted response headers and never credential headers or trace output. Chat responses are bounded to 64,000 visible characters, and session history retains at most 20 messages and 24,000 characters.

Temperature and streaming are not exposed unless catalogue and implementation support are explicitly validated. The modern templates do not use legacy `max_tokens`.

## 10. Model Router test

Select **Model Router**. The request uses EUS2 `model-router` with a larger default allowance:

```json
{
  "model": "model-router",
  "messages": [
    {"role": "user", "content": "Reply in exactly one short sentence: Why is an API gateway useful?"}
  ],
  "max_completion_tokens": 1000
}
```

The requested deployment is `model-router`, but the router may select another model. When supplied, the app reports `x-model-router-selected-model`, routing mode, fallback status, and `x-ms-served-model`. Reasoning tokens can consume the completion allowance. A low allowance can cause empty visible output with `finish_reason: length`. Do not calculate cost from the alias alone.

## 11. Image-generation tests

Starter mode defaults and constrains `n` to `1`, with size `1024x1024`, quality `low`, and format `png`.

### EUS2

```json
{
  "model": "gpt-image-2",
  "prompt": "A clean abstract illustration of a university digital learning platform, blue and purple palette, simple geometric forms, no text",
  "size": "1024x1024",
  "quality": "low",
  "output_format": "png",
  "n": 1
}
```

Send to `https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1/images/generations`.

### WUS3

Use the same body with `"model": "gpt-image-1.5"` and send it to `https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1/images/generations`.

The selector prevents cross-region pairing. The service validates the returned image count before decoder work, bounds the response and decoded sizes, verifies the requested PNG/JPEG/WebP container signature before Pillow dispatch, then checks complete decoding, exact dimensions, and pixel count. It displays the image and offers a local download. Base64 is never logged, rendered as text, or exported. Missing/malformed `b64_json` yields a sanitized diagnostic and request ID.

Image cost depends on model, quality, size, and count. Do not infer it from language-token headers. Initially assess image use through request-rate and call quotas.

## 12. Embedding tests

Embeddings are EUS2-only in this catalogue. `text-embedding-3-small` is the recommended first model; `text-embedding-ada-002` is retained only for legacy compatibility.

Single input:

```json
{
  "model": "text-embedding-3-small",
  "input": "CUHK AI API Portal embedding test"
}
```

Multiple input:

```json
{
  "model": "text-embedding-3-small",
  "input": [
    "Digital learning infrastructure",
    "Generative AI for university teaching",
    "Secure API governance"
  ]
}
```

The comparison sends all texts in one request, verifies vector count, indices, dimensions, finite numeric values, a maximum dimension of 4,096, and an aggregate scalar ceiling before copying vectors or calculating similarity. Inputs are bounded to eight texts, 4,000 characters each, and 12,000 combined characters. The full vector is never displayed by default, logged, or exported; users can explicitly reveal only the first and last eight values.

Similarity is a mathematical comparison—not a factual or quality judgment—and must not be interpreted as plagiarism, authorship, intent, or academic misconduct. Do not claim one embedding model is more accurate without authoritative evidence.

## 13. Rate and quota explanation

Current intended Starter presentation:

| Control | Limit | Scope | Typical result |
|---|---|---|---|
| Product call rate | 10 calls per 60 seconds | APIM subscription | HTTP 429 |
| Product call quota | 100 calls per seven days | APIM subscription across product APIs | HTTP 403 |
| Language-model token rate | 250,000 TPM where configured | APIM subscription | HTTP 429 with `Retry-After` |
| Monthly language-model tokens | 5,000,000 | APIM subscription | HTTP 403 |
| Backend capacity | deployment-specific | Foundry quota scope | backend HTTP 429 |

Product call limits and API token limits are separate; the first reached stops usage. A VS Code Agent task can make multiple API requests. A shared Starter subscription must not be used for a synchronized class. Do not assume images are priced through language-token consumption or embeddings emit the same metrics as Chat Completions.

The UI keeps four concepts separate and bounds application-side records to the newest 100 entries:

1. **CUHK APIM allowance:** `x-cuhk-tokens-consumed`, TPM remaining, monthly tokens remaining, and `Retry-After` when returned.
2. **Foundry backend capacity:** separately labeled `x-ratelimit-*` headers.
3. **Response-body usage:** operation-provided token fields.
4. **Application-side session statistics:** local allowlisted metadata; not an APIM counter or billing ledger.

## 14. Error guide

| Symptom | Guidance |
|---|---|
| Missing key / HTTP 401 | Configure a valid APIM subscription key. The app never echoes it. |
| HTTP 400 `max_tokens` unsupported | Use `max_completion_tokens` for modern Chat Completions models. |
| HTTP 400 `unknown_model` | Choose an exact deployment from the selected regional API. |
| HTTP 403 | A call/token quota, product authorization, or another access condition may have been reached. |
| HTTP 404 | Check region and operation path; the specialized operation may not be exposed by the APIM contract. The app does not silently reroute it to chat. |
| HTTP 429 | Distinguish APIM call rate, APIM token rate, and backend deployment capacity; observe `Retry-After`. |
| HTTP 200 with empty output | Inspect `finish_reason` and reasoning usage; increase `max_completion_tokens` when the reason is `length`. |
| Image missing `b64_json` | Record the sanitized request ID; do not paste the whole response or Base64 into support channels. |
| Embedding count/dimension mismatch | Similarity calculation stops with a structured error. |
| Malformed JSON | Record status and request IDs; do not expose trace or credentials. |

If an operation is unavailable, confirm that the APIM API contract exposes it. Do not substitute Chat Completions for a specialized operation.

## 15. Special product request guidance

Use Starter for individual learning and short non-sensitive tests. Teachers, TAs, FYP leaders, research projects, repository-scale agents, sustained applications, and synchronized classes should request a dedicated product. One personal Starter key must not be shared with a class.

No authoritative mailbox, form, or service-desk category is documented in this repository. Use the official ITSC service channel and include:

- request title;
- requestor and unit;
- accountable teacher, supervisor, PI, or service owner;
- course/project and purpose;
- number and type of users;
- requested start/end dates;
- required models and operations;
- expected simultaneous users and calls per minute;
- expected monthly usage or budget ceiling;
- data classification;
- client type;
- required reporting;
- funding/cost ownership;
- approvals;
- operational and backup contacts.

A reusable copy/paste template is available on the app’s **Special Access** page.

## 16. Troubleshooting

1. Confirm `.env` exists locally but is not tracked: `git status --short` must not show it.
2. Confirm the key is an active APIM product subscription key, not a Foundry backend key.
3. Confirm the selected region contains the exact model ID.
4. Confirm the capability uses the selected operation.
5. Confirm base URLs end at `/openai/v1`, without a dated `api-version` query.
6. For 429, inspect the separately labeled CUHK allowance and backend-capacity panels plus `Retry-After`.
7. For support, provide API/product, client, model, operation, approximate time/time zone, HTTP status, sanitized error, APIM request ID, and `x-request-id`. Never provide the key, full trace, Base64 image, embedding vector, source file, or sensitive prompt.
8. If key exposure is suspected, rotate it rather than displaying it.

## 17. Tests

Tests use mocks/fakes and never call live CUHK APIs or require a real key.

```bash
python -m pytest -q
```

Required local checks:

```bash
ruff check .
python -m py_compile app.py get_started.py clients/*.py config/*.py services/*.py utils/*.py scripts/*.py tests/*.py
python scripts/check_dependency_lock.py
python -m pytest -q
```

Run the dependency audit from the installation section in a connected environment. The test suite covers catalogue mappings, fixed URL/header construction, redirect prevention, request/response ceilings, `max_completion_tokens`, response and bounded header parsing, Model Router metadata, image count/signature/size/error paths, embedding dimension and scalar limits, plain-text rendering, bounded session retention, redaction, sanitized exports, user-facing errors, importability, and keyless loopback-only Streamlit startup.

## 18. Manual smoke-test sequence

After mocked tests pass and the required APIM operations are confirmed:

1. Start without a key and confirm all guidance/catalogue pages render.
2. Configure the APIM subscription key locally; never capture it in a screenshot.
3. Test EUS2 `gpt-5.4-mini` Chat Completions with the exact smoke prompt.
4. Test WUS3 `gpt-5.6-sol` with the same prompt.
5. Test EUS2 `model-router` with a 1,000-token allowance and inspect router headers.
6. Generate one low-quality 1024×1024 PNG with EUS2 `gpt-image-2`.
7. Generate one equivalent image with WUS3 `gpt-image-1.5`.
8. Create one EUS2 `text-embedding-3-small` embedding and reveal only the optional sample.
9. Compare the three default texts and verify vector count/dimension before the matrix appears.
10. Download session statistics and verify they contain no key, prompt, Base64, vectors, or source files.

Live testing requires APIM product access plus these exposed operations: regional `/chat/completions`, regional `/images/generations`, and EUS2 `/embeddings`. Responses, realtime, translation, whisper, and transcription remain catalogue-only in this UI.

## 19. Change log

### 2026-08 — Security and resource hardening

- locked the reviewed runtime and development dependency graphs and added lock-consistency and audit guidance;
- restricted outbound requests to exact CUHK APIM origins and operation paths, with redirects disabled and bounded request/response bodies;
- removed the generic Azure key fallback and enforced loopback-only Streamlit binding with private Codespaces forwarding guidance;
- rendered model output as plain text and bounded prompts, visible responses, headers, metadata, exports, chat history, and session statistics;
- validated image cardinality and container signatures before bounded Pillow decoding;
- bounded embedding dimensions and aggregate scalar counts before vector copying or similarity work;
- expanded mocked regression coverage for the hardening controls without live API calls or credentials.

### 2026-08 — EUS2/WUS3 Foundry modernization

- migrated current demonstrations to CUHK Foundry Models EUS2/WUS3 OpenAI v1 base routes;
- replaced duplicated legacy models/endpoints with one typed regional catalogue;
- added capability-aware Chat, Model Router, regional Image Generation, and EUS2 Embeddings pages;
- added safe response-header/usage separation and sanitized session statistics;
- hardened key handling, error output, rotating metadata-only logs, Base64 validation, and vector privacy;
- updated Starter limits, quick-start, special-access, responsible-use, and academic-honesty guidance;
- added mocked automated tests with no live CUHK API dependency.

## License

MIT — see [LICENSE](LICENSE).
