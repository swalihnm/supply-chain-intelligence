# Deployment

## Local (verified working)

```bash
uv run python scripts/stage_deploy.py   # 23 MB of serving artifacts -> build context
docker compose up --build               # dashboard :8501, API :8000/docs
```

Verified on Docker 29.7.2 / Compose v5.5.1:

| Check | Result |
|---|---|
| Both containers healthy | yes |
| `/health` loads artifact tables | 19/19 |
| `/forecast`, `/policy` | correct per-series values |
| `/simulate` (full 30,490-series run) | ~42 s, numbers identical to the local run |
| Dashboard builds Plotly figures | verified by exec, not by HTTP status |
| `Traceback`/`ERROR` in logs | none |

Image 1.4 GB. API ~209 MB resident, dashboard ~76 MB.

> A 200 from `/_stcore/health` proves only that the Streamlit **server** started
> — the script does not run until a websocket connects. That is why the figure
> check above execs a script inside the container instead of trusting the
> status code.

---

## Public hosting

### The gotcha that decides the host

**`deploy/data/artifacts/` is gitignored.** It is 23 MB of derived data that has
no business in version control.

That rules out the obvious approach on repo-connected hosts:

| Host | Builds from | Works out of the box? |
|---|---|---|
| **Fly.io** | your local image | **yes** — gitignore is irrelevant |
| Render | the repo | no — image would ship with no artifacts |
| Railway | the repo | no — same |

On Render or Railway the container *starts*, finds no serving bundle, and shows
the "artifacts not built" error. Not a crash, which makes it easy to
misdiagnose.

### Fly.io (recommended)

`fly.toml` is already written and the `PORT` override it depends on is verified
working in the container.

```bash
# 1. Install flyctl (once)
winget install -e --id Fly.Flyctl

# 2. Log in — opens a browser, so this step is yours
fly auth login

# 3. Make sure artifacts are current
uv run python scripts/stage_deploy.py

# 4. First time only — claims the app name, does not deploy
fly launch --no-deploy --copy-config

# 5. Ship it
fly deploy
```

`fly deploy` builds and pushes the image; expect a few minutes on the first
push for a 1.4 GB image.

Then put the URL it prints at the top of the README, replacing the
"not yet hosted" line.

### Choices baked into `fly.toml`, and why

- **1 GB, not 512 MB.** The dashboard idles at ~76 MB, but the Policy explorer's
  "Run simulation" button re-derives the policy and simulates all 30,490 series.
  That peaks well above 512 MB and would be OOM-killed on the smaller machine.
- **`min_machines_running = 0`.** Scales to zero when idle, which keeps it inside
  the free allowance. The cost is a cold start on the first request after a
  quiet period — acceptable for a portfolio link.
- **`primary_region = "sin"`.** Change to whatever is nearest your audience.
- **Dashboard, not API.** The dashboard is the thing worth linking. To serve the
  API instead, switch the `app` process to `"api"` and `internal_port` to 8000.

### If you would rather use Render or Railway

Un-ignore the artifacts first:

```bash
# remove the deploy/data/ line from .gitignore, then
git add -f deploy/data/artifacts
git commit -m "Ship serving artifacts for repo-connected deploy"
```

This puts 23 MB of binary data in git history permanently. Fly avoids that,
which is why it is the recommendation.

---

## Refreshing a deployment after re-running the pipeline

The serving bundle is a snapshot. After re-running any of the pipeline scripts:

```bash
uv run python scripts/build_serving_bundle.py
uv run python scripts/stage_deploy.py
fly deploy
```
