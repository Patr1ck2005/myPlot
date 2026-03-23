import re
from pathlib import Path

# ===== 配置 =====
INPUT_FOLDER = r"D:\DELL\Documents\Research\exploration\gauge_field"
OUTPUT_SUBFOLDER = "_fixed"
ENCODING = "utf-8"

def fix_math_indent(text, file):

    lines = text.splitlines()
    result = []

    open_line = None
    open_indent = ""

    for i, line in enumerate(lines):

        m = re.match(r"^(\s*)\$\$\s*$", line)

        if m:

            indent = m.group(1)

            if open_line is None:
                open_line = len(result)
                open_indent = indent
                result.append(line)

            else:
                close_indent = indent

                if open_indent != close_indent:
                    print(
                        f"{file}: line {open_line+1} "
                        f"open indent fixed '{open_indent}' -> '{close_indent}'"
                    )

                    result[open_line] = close_indent + "$$"

                result.append(line)

                open_line = None
                open_indent = ""

        else:
            result.append(line)

    return "\n".join(result)


def process():

    input_dir = Path(INPUT_FOLDER)
    output_dir = input_dir / OUTPUT_SUBFOLDER
    output_dir.mkdir(exist_ok=True)

    for md in input_dir.rglob("*.md"):

        if output_dir in md.parents:
            continue

        rel = md.relative_to(input_dir)
        out = output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        text = md.read_text(encoding="utf8")
        fixed = fix_math_indent(text, rel)

        out.write_text(fixed, encoding="utf8")

        print("Processed:", rel)


if __name__ == "__main__":
    process()