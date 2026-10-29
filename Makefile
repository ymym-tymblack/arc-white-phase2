.PHONY: demo-cast demo-cast-headless install test lint help install-gh setup-github

help:
	@echo "Targets:"
	@echo "  install              uv sync --group dev"
	@echo "  test                 run pytest"
	@echo "  lint                 run ruff check"
	@echo "  install-gh           install GitHub CLI globally via Ubuntu/Debian apt"
	@echo "  setup-github         install gh, sign in once, and configure Git authentication"
	@echo "  demo-cast            record asciinema cast for README opener (needs a TTY)"
	@echo "  demo-cast-headless   build the cast without a TTY (CI / agent sandboxes)"
	@echo "                       SOURCE=. records this checkout instead of GitHub main"

install:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

# Ubuntu/Debian: add GitHub's signed apt repository and install gh system-wide.
# Rewrites the source entry as one line, repairing a malformed entry from manual setup too.
install-gh:
	@command -v apt-get >/dev/null || { echo "install-gh requires Ubuntu/Debian (apt-get)" >&2; exit 1; }
	@command -v curl >/dev/null || { echo "curl is required; install it with: sudo apt-get install curl" >&2; exit 1; }
	@set -eu; \
	  key_tmp=$$(mktemp); \
	  trap 'rm -f "$$key_tmp"' EXIT; \
	  curl -fsSL -o "$$key_tmp" https://cli.github.com/packages/githubcli-archive-keyring.gpg; \
	  sudo install -d -m 0755 /etc/apt/keyrings; \
	  sudo install -m 0644 "$$key_tmp" /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
	  printf 'deb [arch=%s signed-by=%s] %s stable main\n' \
	    "$$(dpkg --print-architecture)" \
	    "/etc/apt/keyrings/githubcli-archive-keyring.gpg" \
	    "https://cli.github.com/packages" \
	    | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null; \
	  sudo apt-get update; \
	  sudo apt-get install -y gh

# Install gh, then authenticate via browser/device flow only if not already logged in.
# gh stores credentials locally; setup-git makes Git use them for HTTPS remotes.
setup-github: install-gh
	@if gh auth status --hostname github.com >/dev/null 2>&1; then \
	  echo "Already authenticated with github.com"; \
	else \
	  gh auth login --hostname github.com --git-protocol https --web; \
	fi
	@gh auth setup-git
	@gh auth status --hostname github.com

# Records a fresh-clone walkthrough cast and renders to GIF.
# Sequence: git clone -> cd -> uv sync -> python estimator.py -> whest validate -> whest run (mini split)
# Output: assets/demo.cast (committed) + assets/demo.gif (committed).
# Requires: asciinema and agg (brew install asciinema agg).
# Clones GitHub main, so it records whatever is already public. To record a release
# that has not merged yet, use demo-cast-headless with SOURCE=. instead.
demo-cast:
	@command -v asciinema >/dev/null || (echo "Install asciinema first: brew install asciinema"; exit 1)
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	@python3 -c "import os; os.close(os.openpty()[0])" 2>/dev/null || ( \
	  echo "No PTY available (CI / agent sandbox) — 'asciinema rec' needs a terminal."; \
	  echo "Run 'make demo-cast-headless' instead."; exit 1)
	@DEMO_DIR=$$(mktemp -d) && \
	  cp scripts/record-demo.sh "$$DEMO_DIR/demo.sh" && \
	  chmod +x "$$DEMO_DIR/demo.sh" && \
	  TERM=xterm-256color FORCE_COLOR=1 asciinema rec --overwrite --idle-time-limit=2 \
	    --window-size 90x36 \
	    --title="whest-starterkit: clone, sync, estimate, validate, score" \
	    --command="bash $$DEMO_DIR/demo.sh $$DEMO_DIR" assets/demo.cast && \
	  rm -rf "$$DEMO_DIR"
	agg --theme monokai --font-size 14 --last-frame-duration 5 \
	  --cols 90 --rows 36 assets/demo.cast assets/demo.gif

# Headless fallback for environments without a PTY (CI / agent sandboxes), where
# 'asciinema rec' can't run. Runs the same sequence, captures real output, strips
# transient spinner frames, assembles assets/demo.cast + renders assets/demo.gif.
# Requires: agg. Keep the sequence in sync with scripts/record-demo.sh.
#
# SOURCE=<path> clones a local checkout instead of GitHub main — the only way to record a
# correct asset for a release that has not merged yet. `git clone` of a path takes committed
# state only, so commit first. Resolved to an absolute path here because the recorder clones
# from a scratch tempdir, where a relative path would not point at this repo.
#   make demo-cast-headless SOURCE=.
SOURCE ?=
demo-cast-headless:
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	python3 scripts/record-demo-headless.py $(if $(SOURCE),--source $(abspath $(SOURCE)))
