#!/usr/bin/env python3
"""
Apply patches to CBBpy installation in virtual environment.

This script modifies the installed CBBpy package to fix NCAA.com API changes.
Run this after creating a new virtual environment or updating CBBpy.

Patches Applied:
1. isConferenceGame - Use .get() to handle missing field
2. Shot chart text - Use .get() to handle missing 'text' field
3. Fuzzy matching - Add None check when team list is empty

Usage:
    python scripts/patch_cbbpy_venv.py

The script will:
- Locate CBBpy in your virtual environment
- Apply all necessary patches
- Clear Python bytecode cache
- Verify patches were applied

This script is safe to run multiple times - it will skip already-patched code.
"""

import sys
from pathlib import Path
import shutil

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_status(message, status='info'):
    """Print colored status message"""
    if status == 'success':
        print(f"{GREEN}✓{RESET} {message}")
    elif status == 'warning':
        print(f"{YELLOW}⚠{RESET} {message}")
    elif status == 'error':
        print(f"{RED}✗{RESET} {message}")
    else:
        print(f"{BLUE}•{RESET} {message}")


def find_cbbpy_utils():
    """Find cbbpy_utils.py in the virtual environment"""
    try:
        import cbbpy.utils.cbbpy_utils as cbbpy_utils
        utils_path = Path(cbbpy_utils.__file__)
        return utils_path
    except ImportError:
        return None


def backup_file(file_path):
    """Create backup of file before patching"""
    backup_path = file_path.with_suffix('.py.backup')

    if backup_path.exists():
        print_status(f"Backup already exists: {backup_path.name}", 'info')
        return backup_path

    shutil.copy2(file_path, backup_path)
    print_status(f"Created backup: {backup_path.name}", 'success')
    return backup_path


def apply_patch_1_isconferencegame(content):
    """Patch 1: Fix KeyError for 'isConferenceGame'"""
    # Find the line with direct access
    old_line = 'is_conference = more_info["isConferenceGame"]'
    new_line = 'is_conference = more_info.get("isConferenceGame", False)  # Patched: NCAA.com removed this field'

    if old_line in content:
        print_status("Applying Patch 1: isConferenceGame KeyError fix", 'info')
        content = content.replace(old_line, new_line)
        print_status("Patch 1 applied", 'success')
        return content, True
    elif new_line in content:
        print_status("Patch 1 already applied", 'success')
        return content, False
    else:
        print_status("Patch 1: Could not find target line (CBBpy may have changed)", 'warning')
        return content, False


def apply_patch_2_shot_chart(content):
    """Patch 2: Fix KeyError for shot chart 'text' field"""
    old_line = 'shotdescs = [x["text"] for x in chart]'
    new_line = 'shotdescs = [x.get("text", "") for x in chart]  # Patched: handle missing \'text\' field'

    if old_line in content:
        print_status("Applying Patch 2: Shot chart text field fix", 'info')
        content = content.replace(old_line, new_line)
        print_status("Patch 2 applied", 'success')
        return content, True
    elif new_line in content:
        print_status("Patch 2 already applied", 'success')
        return content, False
    else:
        print_status("Patch 2: Could not find target line (CBBpy may have changed)", 'warning')
        return content, False


def apply_patch_3_fuzzy_match(content):
    """Patch 3: Add None check for fuzzy matching"""
    # This is a multi-line patch
    old_code = """    best_match, score, _ = process.extractOne(
        team, id_map['location'].tolist(), scorer=fuzz.token_sort_ratio
    )"""

    new_code = """    result = process.extractOne(
        team, id_map['location'].tolist(), scorer=fuzz.token_sort_ratio
    )
    if result is None:  # Patched: handle empty team list
        raise ValueError(f"No teams found for season {season}. Team list may be empty.")
    best_match, score, _ = result"""

    if old_code in content:
        print_status("Applying Patch 3: Fuzzy matching None check", 'info')
        content = content.replace(old_code, new_code)
        print_status("Patch 3 applied", 'success')
        return content, True
    elif "if result is None:  # Patched: handle empty team list" in content:
        print_status("Patch 3 already applied", 'success')
        return content, False
    else:
        print_status("Patch 3: Could not find target code (CBBpy may have changed)", 'warning')
        return content, False


def clear_pycache(utils_path):
    """Clear Python bytecode cache"""
    pycache_dir = utils_path.parent / '__pycache__'

    if not pycache_dir.exists():
        print_status("No __pycache__ directory found", 'info')
        return

    deleted_count = 0
    for pyc_file in pycache_dir.glob('cbbpy_utils.*.pyc'):
        try:
            pyc_file.unlink()
            deleted_count += 1
        except Exception as e:
            print_status(f"Could not delete {pyc_file.name}: {e}", 'warning')

    if deleted_count > 0:
        print_status(f"Cleared {deleted_count} bytecode cache file(s)", 'success')
    else:
        print_status("No bytecode cache to clear", 'info')


def main():
    print("\n" + "=" * 60)
    print("CBBpy Virtual Environment Patcher")
    print("=" * 60 + "\n")

    # Find CBBpy installation
    print_status("Locating CBBpy installation...", 'info')
    utils_path = find_cbbpy_utils()

    if utils_path is None:
        print_status("CBBpy not found. Install with: pip install cbbpy", 'error')
        sys.exit(1)

    print_status(f"Found: {utils_path}", 'success')

    # Create backup
    print_status("\nCreating backup...", 'info')
    backup_path = backup_file(utils_path)

    # Read file
    print_status("\nReading cbbpy_utils.py...", 'info')
    with open(utils_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply patches
    print_status("\nApplying patches...", 'info')
    print("-" * 60)

    modified = False

    content, changed = apply_patch_1_isconferencegame(content)
    modified = modified or changed

    content, changed = apply_patch_2_shot_chart(content)
    modified = modified or changed

    content, changed = apply_patch_3_fuzzy_match(content)
    modified = modified or changed

    print("-" * 60)

    # Write patched file
    if modified:
        print_status("\nWriting patched file...", 'info')
        with open(utils_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print_status("File updated successfully", 'success')
    else:
        print_status("\nNo changes needed - all patches already applied", 'success')

    # Clear cache
    print_status("\nClearing Python bytecode cache...", 'info')
    clear_pycache(utils_path)

    # Summary
    print("\n" + "=" * 60)
    print_status("CBBpy patching complete!", 'success')
    print("=" * 60)

    print("\nPatched file:", utils_path)
    print("Backup file: ", backup_path)

    print("\nNext steps:")
    print("  1. Test CBBpy with: python -m src.data_sources.cbbpy_enhanced")
    print("  2. Fetch 2026 data with: python scripts/fetch_all_data.py --year 2026 --sources cbbpy --teams Duke")

    # Also ensure 2026 season exists
    print("\n" + "-" * 60)
    print_status("Ensuring season 2026 exists in team map...", 'info')

    try:
        from src.data_sources.cbbpy_patches import ensure_season_in_team_map

        if ensure_season_in_team_map(2026):
            print_status("Season 2026 ready", 'success')
        else:
            print_status("Could not add season 2026", 'warning')

        # Also add 2027 for future
        if ensure_season_in_team_map(2027):
            print_status("Season 2027 ready (future)", 'success')

    except Exception as e:
        print_status(f"Could not update team map: {e}", 'warning')
        print("  Run manually: python -m src.data_sources.cbbpy_patches")

    print()


if __name__ == '__main__':
    main()
