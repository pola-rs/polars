import os
import re


def remove_meta_redirects(directory):
    meta_redirect_pattern = re.compile(r'\s*<meta\s+http-equiv="refresh"[^>]+>', re.IGNORECASE)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Remove meta refresh tags
                new_content = re.sub(meta_redirect_pattern, "", content)
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Removed meta refresh from: {file_path}")


if __name__ == "__main__":
    remove_meta_redirects(os.getcwd())
