#!/usr/bin/env python3
import os
import sys
import curses
import argparse
import time
import tiktoken
import pathspec
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global cache for directory scans: path -> either list of items or Future.
scan_cache = {}
cache_lock = threading.Lock()

# Global executor for asynchronous scanning.
scan_executor = ThreadPoolExecutor(max_workers=4)

# ---------------- Utility Functions ----------------

def find_git_root(starting_dir):
    current = os.path.abspath(starting_dir)
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, '.git')):
            return current
        current = os.path.dirname(current)
    return starting_dir

def load_gitignore(git_root):
    gitignore_path = os.path.join(git_root, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
            patterns = f.readlines()
        spec = pathspec.PathSpec.from_lines('gitwildmatch', patterns)
    else:
        spec = pathspec.PathSpec.from_lines('gitwildmatch', [])
    return spec

def is_binary(filepath):
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
    except Exception:
        return True
    return False

def get_encoder(encoding_name=None, model_name=None):
    if encoding_name:
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            pass

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            pass

    # Fallbacks
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.encoding_for_model("gpt-4o")

def count_tokens_in_file(filepath, encoder):
    if is_binary(filepath):
        return 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception:
        return 0

def aggregate_tokens_in_dir(path, encoder, git_spec, repo_root):
    total = 0
    try:
        for entry in os.scandir(path):
            rel_path = os.path.relpath(entry.path, repo_root)
            if git_spec.match_file(rel_path) or entry.name == '.git':
                continue
            if entry.is_file(follow_symlinks=False):
                total += count_tokens_in_file(entry.path, encoder)
            elif entry.is_dir(follow_symlinks=False):
                total += aggregate_tokens_in_dir(entry.path, encoder, git_spec, repo_root)
    except PermissionError:
        pass
    return total

def scan_directory(path, encoder, git_spec, repo_root):
    """
    Scans the directory non-recursively. For directories, recursively
    aggregates token counts.
    """
    items = []
    with ThreadPoolExecutor() as executor:
        futures = {}
        try:
            for entry in os.scandir(path):
                rel_path = os.path.relpath(entry.path, repo_root)
                # Skip if the file/directory is in gitignore
                if git_spec.match_file(rel_path) or entry.name == '.git':
                    continue

                if entry.is_file(follow_symlinks=False):
                    future = executor.submit(count_tokens_in_file, entry.path, encoder)
                    futures[future] = entry
                elif entry.is_dir(follow_symlinks=False):
                    future = executor.submit(aggregate_tokens_in_dir, entry.path, encoder, git_spec, repo_root)
                    futures[future] = entry
        except PermissionError:
            pass

        for future in as_completed(futures):
            entry = futures[future]
            try:
                tokens = future.result()
            except Exception:
                tokens = 0

            if entry.is_file(follow_symlinks=False):
                items.append({
                    'name': entry.name,
                    'path': entry.path,
                    'is_dir': False,
                    'token_count': tokens
                })
            elif entry.is_dir(follow_symlinks=False):
                # Only include directories with non-zero token count
                if tokens > 0:
                    items.append({
                        'name': entry.name + os.sep,
                        'path': entry.path,
                        'is_dir': True,
                        'token_count': tokens
                    })

    items.sort(key=lambda x: x['token_count'], reverse=True)
    return items

def cached_scan_directory(path, encoder, git_spec, repo_root):
    """
    Returns a list of items if the directory scan is complete.
    If a scan is in progress, returns None.
    Otherwise, starts an asynchronous scan.
    """
    with cache_lock:
        if path in scan_cache:
            result = scan_cache[path]
            if isinstance(result, list):
                return result
            else:  # It's a Future.
                if result.done():
                    try:
                        items = result.result()
                    except Exception:
                        items = []
                    scan_cache[path] = items
                    return items
                else:
                    return None  # Still scanning.
        else:
            # Start scanning asynchronously.
            future = scan_executor.submit(scan_directory, path, encoder, git_spec, repo_root)
            scan_cache[path] = future
            return None

def progress_bar(percentage, bar_width):
    """Return a string representing a progress bar for the given percentage."""
    fill = int(percentage * bar_width)
    bar = "[" + "#" * fill + "-" * (bar_width - fill) + "]"
    return bar

# ---------------- TUI Functions ----------------

def draw_menu(stdscr, items, selected_idx, current_path, offset, scanning):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    header = f"Tokdu - {current_path} (Enter: open, Backspace: up, q: quit)"
    stdscr.addstr(0, 0, header.ljust(width), curses.A_REVERSE)
    max_items = height - 2

    # If scanning is in progress, show an indicator.
    if scanning:
        scan_msg = "Scanning directory... please wait."
        stdscr.addstr(1, 0, scan_msg.ljust(width), curses.A_BLINK)

    if items is None:
        stdscr.refresh()
        return offset

    # Calculate total tokens in the current directory to compute percentages.
    total_tokens = sum(item['token_count'] for item in items)
    if total_tokens == 0:
        total_tokens = 1  # Avoid division by zero.

    if selected_idx < offset:
        offset = selected_idx
    elif selected_idx >= offset + max_items:
        offset = selected_idx - max_items + 1

    # Calculate progress bar width as 20% of available width.
    bar_width = max(10, int(width * 0.2))

    for idx, item in enumerate(items[offset:offset+max_items]):
        perc = item['token_count'] / total_tokens
        bar = progress_bar(perc, bar_width)
        perc_text = f"{perc*100:3.0f}%"
        # Build the line: token count, progress bar, percentage, then file/directory name.
        line = f"{item['token_count']:10}  {bar} {perc_text}  {item['name']}"
        # Ensure the line does not exceed the available width.
        line = line[:width-1]
        if idx + offset == selected_idx:
            stdscr.addstr(idx + 1, 0, line.ljust(width), curses.A_STANDOUT)
        else:
            stdscr.addstr(idx + 1, 0, line.ljust(width))
    stdscr.refresh()
    return offset

def tui(stdscr, start_path, encoding_name, model_name):
    curses.curs_set(0)
    stdscr.nodelay(True)  # Enable non-blocking input.
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)

    encoder = get_encoder(encoding_name, model_name)
    repo_root = find_git_root(start_path)
    git_spec = load_gitignore(repo_root)

    # Save the absolute starting directory.
    root_dir = os.path.abspath(start_path)
    current_path = root_dir
    history = []  # For back navigation.

    selected_idx = 0
    offset = 0

    while True:
        items = cached_scan_directory(current_path, encoder, git_spec, repo_root)
        scanning = items is None

        height, width = stdscr.getmaxyx()
        max_items = height - 2
        offset = draw_menu(stdscr, items, selected_idx, current_path, offset, scanning)

        key = stdscr.getch()
        if key == -1:
            time.sleep(0.1)
            continue

        if key in (ord('q'), ord('Q')):
            break
        elif key in (curses.KEY_UP, ord('k')):
            if selected_idx > 0:
                selected_idx -= 1
        elif key in (curses.KEY_DOWN, ord('j')):
            if items and selected_idx < len(items) - 1:
                selected_idx += 1
        elif key == curses.KEY_NPAGE:  # Page Down
            if items:
                selected_idx = min(len(items) - 1, selected_idx + max_items)
        elif key == curses.KEY_PPAGE:  # Page Up
            selected_idx = max(0, selected_idx - max_items)
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter key
            if items:
                chosen = items[selected_idx]
                if chosen['is_dir']:
                    history.append(current_path)
                    current_path = os.path.abspath(chosen['path'])
                    selected_idx = 0
                    offset = 0
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            # Prevent navigating above the starting directory.
            if current_path == root_dir:
                curses.flash()
            else:
                parent = os.path.dirname(current_path)
                if os.path.commonpath([root_dir, parent]) != root_dir:
                    curses.flash()
                else:
                    history.append(current_path)
                    current_path = parent
                    selected_idx = 0
                    offset = 0

# ---------------- Main Entry ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokdu: A token counting TUI tool (respects .gitignore, skips binary files)."
    )
    parser.add_argument(
        "directory", nargs="?", default=".",
        help="Directory to start in (default: current directory)"
    )
    parser.add_argument(
        "--encoding", "-e", dest="encoding_name",
        help="Tiktoken encoding name (e.g., 'cl100k_base', 'o200k_base')"
    )
    parser.add_argument(
        "--model", "-m", dest="model_name",
        help="Model name for tokenization (e.g., 'gpt-3.5-turbo', 'gpt-4o')"
    )
    args = parser.parse_args()

    try:
        curses.wrapper(tui, args.directory, args.encoding_name, args.model_name)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.exit(str(e))

if __name__ == "__main__":
    main()
