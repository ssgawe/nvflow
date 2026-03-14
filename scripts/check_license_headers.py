# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Check that Python source files have the required NVIDIA Apache 2.0 license header."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Canonical header: exact lines files must start with (year on line 1 is flexible).
LICENSE_HEADER_LINES = [
    "# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "#     http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
    "#",
]
COPYRIGHT_LINE_PATTERN = re.compile(
    r"^# Copyright \(c\) \d{4}, NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.$"
)

SKIP_DIRS = {".venv", "__pycache__", ".git"}
SKIP_FILES = {"_version.py"}


def should_skip(path: Path, root: Path) -> bool:
    if path.name in SKIP_FILES:
        return True
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIRS for part in rel.parts)


def check_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Cannot read: {e}"
    lines = text.splitlines()
    if len(lines) < len(LICENSE_HEADER_LINES):
        return "File too short to contain license header"
    # Allow optional shebang and blank lines before the header (e.g. #!/usr/bin/env python3)
    start = 0
    for i in range(min(5, len(lines))):
        line = lines[i]
        if COPYRIGHT_LINE_PATTERN.match(line):
            start = i
            break
        if line.strip() and not line.strip().startswith("#!"):
            return f"Line {i + 1}: expected copyright or optional shebang, got {line!r}"
    else:
        return "First line must be: # Copyright (c) YYYY, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    if start + len(LICENSE_HEADER_LINES) > len(lines):
        return "File too short to contain full license header"
    for j, expected in enumerate(LICENSE_HEADER_LINES):
        actual = lines[start + j]
        if j == 0:
            if not COPYRIGHT_LINE_PATTERN.match(actual):
                return "Copyright line must be: # Copyright (c) YYYY, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
        elif actual != expected:
            return f"Line {start + j + 1}: expected {expected!r}, got {actual!r}"
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        paths = [Path(p).resolve() for p in sys.argv[1:] if p.endswith(".py")]
    else:
        paths = sorted(root.rglob("*.py"))
    paths = [p for p in paths if p.is_file() and not should_skip(p, root)]
    errors = []
    for p in paths:
        err = check_file(p)
        if err is not None:
            errors.append((p, err))
    if errors:
        for path, msg in errors:
            print(path.relative_to(root), msg, file=sys.stderr, sep=": ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
