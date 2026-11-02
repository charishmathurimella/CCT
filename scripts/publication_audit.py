"""Pre-publication checks for a synthetic AI engineering portfolio.

Run at the repository root: python scripts/publication_audit.py
The scanner is a guardrail, not a guarantee that no sensitive material exists.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("intent-classifier", "document-rag", "conversation-regression")
REQUIRED = {
    "intent-classifier": ("README.md", "requirements.txt", "train.py", "validate.py", "tests/test_train.py"),
    "document-rag": ("README.md", "requirements.txt", "rag.py", "evaluate.py", "tests/test_rag.py"),
    "conversation-regression": ("README.md", "requirements.txt", "router.py", "run_regression.py", "tests/test_router.py"),
}
FORBIDDEN_FILES = (".env", "id_rsa", "id_ed25519", "credentials.json", "service-account.json")
SUSPICIOUS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "openai api key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic credential assignment": re.compile(r"(?im)^\s*(?:api_key|secret_key|access_token|password)\s*=\s*['\"][^'\"]{12,}['\"]\s*$"),
}
SCAN_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml", ".toml", ".ini"}


def tracked_files():
    try:
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        return [ROOT / name for name in output.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def audit():
    errors = []
    for project, paths in REQUIRED.items():
        for relative in paths:
            if not (ROOT / project / relative).is_file():
                errors.append(f"Missing required file: {project}/{relative}")
    for path in tracked_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name.lower() in FORBIDDEN_FILES:
            errors.append(f"Sensitive filename: {relative}")
        if path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            errors.append(f"Could not inspect: {relative}")
            continue
        for label, pattern in SUSPICIOUS.items():
            if pattern.search(text):
                errors.append(f"Possible {label}: {relative}")
    if errors:
        print("Publication audit FAILED:")
        for error in errors:
            print(" -", error)
        return 1
    print("Publication audit PASSED: required project files found; no matching credential patterns.")
    print("Manual review of git history, ownership, privacy, licenses, and documentation is still required.")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
