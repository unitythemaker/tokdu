#!/usr/bin/env python3
import os
import sys
import curses
import argparse
import time
import pathspec
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config

# Global cache for directory scans: path -> either list of items or Future.
scan_cache = {}
cache_lock = threading.Lock()

# Global executor for asynchronous scanning.
scan_executor = ThreadPoolExecutor(max_workers=4)

# Flag to track whether curses has been initialized
curses_initialized = False

# ---------------- Utility Functions ----------------

def clean_exit_with_message(message, code=1):
    """
    Safely cleans up curses and exits with an error message.

    Args:
        message: The error message to display
    """
    global curses_initialized
    if curses_initialized:
        try:
            curses.endwin()
        except Exception:
            pass
    print(f"{message}")
    sys.exit(code)

def find_git_root(starting_dir):
    current = os.path.abspath(starting_dir)
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, '.git')):
            return current
        current = os.path.dirname(current)
    return starting_dir

def load_gitignore(git_root, current_dir):
    """
    Load .gitignore files from git_root up to and including current_dir.
    Patterns in deeper directories override those in parent directories,
    matching Git's actual behavior.
    """
    git_root = os.path.abspath(git_root)
    current_dir = os.path.abspath(current_dir)

    # Collect all directories from git_root to current_dir
    directories = []
    path = current_dir
    while os.path.commonpath([path, git_root]) == git_root:
        directories.append(path)
        if path == git_root:
            break
        path = os.path.dirname(path)

    # Process directories from root to current (so deeper patterns override parent patterns)
    directories.reverse()

    # Collect all patterns
    all_patterns = []
    for directory in directories:
        gitignore_path = os.path.join(directory, '.gitignore')
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                    patterns = f.readlines()
                    all_patterns.extend(patterns)
            except Exception:
                pass  # Skip if we can't read the file

    # Create spec with all collected patterns
    spec = pathspec.PathSpec.from_lines('gitwildmatch', all_patterns)
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

def get_encoder(encoding_name=None, model_name=None, tokenizer_type=None):
    """
    Get a tokenizer based on the specified parameters.

    Args:
        encoding_name: The name of a tiktoken encoding
        model_name: The name of a model (OpenAI, Gemini, or Anthropic)
        tokenizer_type: The type of tokenizer to use ('tiktoken', 'gemini', or 'anthropic')

    Returns:
        tuple: (encoder, tokenizer_info_string)
    """
    # Load defaults from config if values not provided
    if tokenizer_type is None:
        tokenizer_type = config.get_config_value('tokenizer', 'type', 'tiktoken')

    # If neither encoding_name nor model_name is provided, load from config
    # respecting the mutual exclusivity (one of them will be empty)
    if encoding_name is None and model_name is None:
        encoding_name = config.get_config_value('tokenizer', 'encoding', '')
        model_name = config.get_config_value('tokenizer', 'model', 'gpt-4o')

        # Ensure empty string is treated as None
        if encoding_name == '':
            encoding_name = None
        if model_name == '':
            model_name = None

    # Try to use Anthropic tokenizer if specified
    if tokenizer_type == 'anthropic':
        try:
            import anthropic

            # Get API key from config or environment variable
            anthropic_api_key = config.get_config_value('tokenizer', 'anthropic_api_key', '') or os.environ.get('ANTHROPIC_API_KEY', '')

            if not anthropic_api_key:
                print("Warning: Anthropic tokenization requires an API key.")
                print("Set it via ANTHROPIC_API_KEY environment variable or configuration.")

                # Ask for permission to use the API
                permission = input("Anthropic token counting requires API calls. Do you want to proceed? (y/n): ")
                if permission.lower() not in ('y', 'yes'):
                    print("Anthropic tokenization aborted.")
                    sys.exit(1)
                else:
                    # Ask for API key
                    anthropic_api_key = input("Enter your Anthropic API key: ")
                    save_key = input("Save this key to configuration? (y/n): ")
                    if save_key.lower() in ('y', 'yes'):
                        config.set_config_value('tokenizer', 'anthropic_api_key', anthropic_api_key)

            if anthropic_api_key and tokenizer_type == 'anthropic':
                # Adapter class to provide a compatible interface
                class AnthropicTokenizerAdapter:
                    def __init__(self, model_name, api_key):
                        self.client = anthropic.Anthropic(api_key=api_key)
                        self.model = model_name if model_name else "claude-3-haiku-20240307"

                        # Retry parameters
                        self.max_retries = 5
                        self.base_backoff = 1.0  # Base backoff in seconds

                        # Error tracking
                        self.error_count = 0
                        self.max_errors = 3
                        self.error_reset_time = 300  # 5 minutes
                        self.last_error_time = 0

                        # Rate limiting parameters
                        self.rate_limited = False
                        self.last_rate_limit_warning = 0
                        self.warning_interval = 60  # Only warn about rate limits once per minute

                    def encode(self, text):
                        # Don't make API calls for empty texts
                        if not text.strip():
                            return []

                        # Reset error count if it's been a while since the last error
                        current_time = time.time()
                        if current_time - self.last_error_time > self.error_reset_time:
                            self.error_count = 0

                        # Retry logic with exponential backoff
                        retries = 0
                        backoff = self.base_backoff

                        while retries <= self.max_retries:
                            try:
                                # Call the token counting API
                                response = self.client.messages.count_tokens(
                                    model=self.model,
                                    messages=[
                                        {"role": "user", "content": text}
                                    ]
                                )
                                # Successful response, reset rate limit flag
                                self.rate_limited = False
                                self.error_count = 0
                                # Create a list of the appropriate length for counting
                                return [0] * response.input_tokens

                            except anthropic.RateLimitError:
                                # Special handling for rate limit errors
                                retries += 1
                                if retries <= self.max_retries:
                                    # Sleep with exponential backoff before retrying
                                    time.sleep(backoff)
                                    backoff *= 2  # Exponential backoff
                                else:
                                    # Out of retries for rate limiting
                                    current_time = time.time()
                                    if current_time - self.last_rate_limit_warning > self.warning_interval:
                                        print(f"Anthropic API rate limit reached after {retries} retries. Token counts will be temporarily unavailable.")
                                        self.last_rate_limit_warning = current_time
                                    self.rate_limited = True
                                    return []

                            except Exception as e:
                                # Handle other errors
                                self.error_count += 1
                                self.last_error_time = current_time

                                # If we've had too many errors, exit the program
                                if self.error_count >= self.max_errors:
                                    clean_exit_with_message(f"Anthropic API error threshold exceeded ({self.error_count} errors). Last error: {e}")

                                # If we still have retries, try again
                                retries += 1
                                if retries <= self.max_retries:
                                    time.sleep(backoff)
                                    backoff *= 2  # Exponential backoff
                                else:
                                    # Out of retries, exit with error
                                    clean_exit_with_message(f"Error counting tokens with Anthropic API after {retries} attempts: {e}")

                        # Should not reach here, but just in case
                        return []

                actual_model = model_name if model_name else "claude-3-haiku-20240307"
                tokenizer_info = f"Anthropic tokenizer (model: {actual_model})"
                return AnthropicTokenizerAdapter(model_name, anthropic_api_key), tokenizer_info

        except ImportError:
            print("Error: Anthropic tokenization requested but anthropic package not installed.")
            print("Install with: pip install anthropic")
            sys.exit(1)

    # Try to use Gemini tokenizer if specified
    if tokenizer_type == 'gemini':
        try:
            from vertexai.preview import tokenization
            # Adapter class to provide a compatible interface with tiktoken
            class GeminiTokenizerAdapter:
                def __init__(self, model_name):
                    self.tokenizer = tokenization.get_tokenizer_for_model(model_name)
                    self.model_name = model_name

                def encode(self, text):
                    # Return a list-like object with a length (just for token counting)
                    result = self.tokenizer.count_tokens(text)
                    # Create a list of the appropriate length for counting
                    return [0] * result.total_tokens

            gemini_model = model_name if model_name else "gemini-2.0-flash"
            tokenizer_info = f"Gemini tokenizer (model: {gemini_model})"
            return GeminiTokenizerAdapter(gemini_model), tokenizer_info
        except ImportError:
            print("Error: Gemini tokenization requested but vertexai package not installed.")
            print("Install with: pip install google-cloud-aiplatform[tokenization]")
            sys.exit(1)

    # Use tiktoken if specified or as fallback
    try:
        import tiktoken

        if encoding_name:
            try:
                tokenizer_info = f"tiktoken (encoding: {encoding_name})"
                return tiktoken.get_encoding(encoding_name), tokenizer_info
            except Exception:
                pass

        if model_name and tokenizer_type != 'gemini':
            try:
                tokenizer_info = f"tiktoken (model: {model_name})"
                return tiktoken.encoding_for_model(model_name), tokenizer_info
            except Exception:
                pass

        # Fallbacks
        try:
            tokenizer_info = "tiktoken (encoding: o200k_base)"
            return tiktoken.get_encoding("o200k_base"), tokenizer_info
        except Exception:
            tokenizer_info = "tiktoken (model: gpt-4o)"
            return tiktoken.encoding_for_model("gpt-4o"), tokenizer_info

    except ImportError:
        print("Error: tiktoken not installed.")
        print("Install with: pip install tiktoken")
        sys.exit(1)

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

def draw_menu(stdscr, items, selected_idx, current_path, offset, scanning, tokenizer_info):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    header = f"Tokdu - {current_path} (Enter: open, Backspace: up, q: quit)"
    stdscr.addstr(0, 0, header.ljust(width), curses.A_REVERSE)

    # Display tokenizer info on the second line
    tokenizer_line = f"Using: {tokenizer_info}".ljust(width)
    stdscr.addstr(1, 0, tokenizer_line, curses.A_BOLD)

    max_items = height - 3  # Reduce by one more line for tokenizer info

    # If scanning is in progress, show an indicator.
    if scanning:
        scan_msg = "Scanning directory... please wait."
        stdscr.addstr(2, 0, scan_msg.ljust(width), curses.A_BLINK)

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
            stdscr.addstr(idx + 2, 0, line.ljust(width), curses.A_STANDOUT)  # Start from line 2 now
        else:
            stdscr.addstr(idx + 2, 0, line.ljust(width))
    stdscr.refresh()
    return offset

def tui(stdscr, start_path, encoder_and_info, repo_root):
    global curses_initialized
    curses_initialized = True

    # Unpack encoder and info
    encoder, tokenizer_info = encoder_and_info

    curses.curs_set(0)
    stdscr.nodelay(True)  # Enable non-blocking input.
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)

    # Save the absolute starting directory.
    root_dir = os.path.abspath(start_path)
    current_path = root_dir
    history = []  # For back navigation.

    selected_idx = 0
    offset = 0
    last_flash_time = 0
    flash_cooldown = 1.0  # Seconds to wait before flashing again
    root_message = ""  # Message to display when at root boundary

    while True:
        # Load or reload gitignore for the current directory
        git_spec = load_gitignore(repo_root, current_path)

        items = cached_scan_directory(current_path, encoder, git_spec, repo_root)
        scanning = items is None

        height, width = stdscr.getmaxyx()
        max_items = height - 3  # Reduced by one for tokenizer info
        offset = draw_menu(stdscr, items, selected_idx, current_path, offset, scanning, tokenizer_info)

        # Display root boundary message if present
        if root_message:
            message_y = height - 1 if height > 3 else 0
            stdscr.addstr(message_y, 0, root_message[:width-1], curses.A_BOLD)
            stdscr.refresh()
            # Auto-clear message after a brief period
            if time.time() - last_flash_time > 2.0:
                root_message = ""

        key = stdscr.getch()
        if key == -1:
            time.sleep(0.1)
            continue

        if key in (ord('q'), ord('Q')):
            break
        elif key in (curses.KEY_UP, ord('k')):
            root_message = ""  # Clear message on navigation
            if selected_idx > 0:
                selected_idx -= 1
        elif key in (curses.KEY_DOWN, ord('j')):
            root_message = ""
            if items and selected_idx < len(items) - 1:
                selected_idx += 1
        elif key == curses.KEY_NPAGE:  # Page Down
            root_message = ""
            if items:
                selected_idx = min(len(items) - 1, selected_idx + max_items)
        elif key == curses.KEY_PPAGE:  # Page Up
            root_message = ""
            selected_idx = max(0, selected_idx - max_items)
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter key
            root_message = ""
            if items:
                chosen = items[selected_idx]
                if chosen['is_dir']:
                    history.append(current_path)
                    current_path = os.path.abspath(chosen['path'])
                    selected_idx = 0
                    offset = 0
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            current_time = time.time()
            # Prevent navigating above the starting directory.
            if current_path == root_dir:
                if current_time - last_flash_time > flash_cooldown:
                    curses.flash()
                    root_message = "Root directory reached. Cannot navigate further up."
                    last_flash_time = current_time
            else:
                parent = os.path.dirname(current_path)
                if os.path.commonpath([root_dir, parent]) != root_dir:
                    if current_time - last_flash_time > flash_cooldown:
                        curses.flash()
                        root_message = "Root directory reached. Cannot navigate further up."
                        last_flash_time = current_time
                else:
                    root_message = ""
                    history.append(current_path)
                    current_path = parent
                    selected_idx = 0
                    offset = 0

# ---------------- Configuration Commands ----------------

def handle_config_command(args):
    """Handle the config subcommand."""
    if args.show:
        config.print_config()
        return

    # Set tokenizer type
    if args.tokenizer_type:
        config.set_config_value('tokenizer', 'type', args.tokenizer_type)
        print(f"Default tokenizer type set to: {args.tokenizer_type}")

    # Set model (will clear encoding)
    if args.model_name:
        cleared = config.set_config_value('tokenizer', 'model', args.model_name)
        print(f"Default model set to: {args.model_name}")
        if cleared:
            print(f"Note: '{cleared}' setting has been cleared since it's mutually exclusive with 'model'")

    # Set encoding (will clear model)
    if args.encoding_name:
        cleared = config.set_config_value('tokenizer', 'encoding', args.encoding_name)
        print(f"Default encoding set to: {args.encoding_name}")
        if cleared:
            print(f"Note: '{cleared}' setting has been cleared since it's mutually exclusive with 'encoding'")

    # If no options were provided, show the current configuration
    if not (args.tokenizer_type or args.model_name or args.encoding_name):
        config.print_config()

# ---------------- Main Entry ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokdu: A token counting TUI tool (respects .gitignore, skips binary files)."
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Default command (scan)
    scan_parser = subparsers.add_parser('scan', help='Scan and analyze token usage (default command)')
    scan_parser.add_argument(
        "directory", nargs="?", default=".",
        help="Directory to start in (default: current directory)"
    )

    tokenizer_group = scan_parser.add_argument_group('Tokenizer Options')
    tokenizer_group.add_argument(
        "--encoding", "-e", dest="encoding_name",
        help="Tiktoken encoding name (e.g., 'cl100k_base', 'o200k_base')"
    )
    tokenizer_group.add_argument(
        "--model", "-m", dest="model_name",
        help="Model name for tokenization (e.g., 'gpt-3.5-turbo', 'gpt-4o', 'gemini-1.5-flash-001')"
    )
    tokenizer_group.add_argument(
        "--tokenizer", "-t", dest="tokenizer_type", choices=['tiktoken', 'gemini', 'anthropic'],
        help="Tokenizer backend to use (default from config or 'tiktoken')"
    )

    # Config command
    config_parser = subparsers.add_parser('config', help='View or set configuration options')
    config_parser.add_argument(
        "--show", action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--tokenizer", dest="tokenizer_type", choices=['tiktoken', 'gemini', 'anthropic'],
        help="Set default tokenizer type"
    )
    config_parser.add_argument(
        "--model", "-m", dest="model_name",
        help="Set default model name"
    )
    config_parser.add_argument(
        "--encoding", "-e", dest="encoding_name",
        help="Set default tiktoken encoding name"
    )

    # Also support directory as a positional argument without subcommand for backward compatibility
    parser.add_argument(
        "directory_compat", nargs="?", default=None,
        help=argparse.SUPPRESS  # Hidden parameter for backward compatibility
    )

    # Add tokenizer options to the main parser for backward compatibility
    parser.add_argument(
        "--encoding", "-e", dest="encoding_name_compat",
        help=argparse.SUPPRESS  # Hidden parameter for backward compatibility
    )
    parser.add_argument(
        "--model", "-m", dest="model_name_compat",
        help=argparse.SUPPRESS  # Hidden parameter for backward compatibility
    )
    parser.add_argument(
        "--tokenizer", "-t", dest="tokenizer_type_compat", choices=['tiktoken', 'gemini', 'anthropic'],
        help=argparse.SUPPRESS  # Hidden parameter for backward compatibility
    )

    args = parser.parse_args()

    # Handle backward compatibility for no subcommand
    if not args.command and (args.directory_compat is not None or
                            args.encoding_name_compat is not None or
                            args.model_name_compat is not None or
                            args.tokenizer_type_compat is not None):
        args.command = 'scan'
        args.directory = args.directory_compat or "."
        args.encoding_name = args.encoding_name_compat
        args.model_name = args.model_name_compat
        args.tokenizer_type = args.tokenizer_type_compat

    # Handle the command
    if args.command == 'config':
        handle_config_command(args)
    else:  # Default to 'scan' command if no command provided
        try:
            directory = args.directory if hasattr(args, 'directory') else "."
            encoding_name = getattr(args, 'encoding_name', None)
            model_name = getattr(args, 'model_name', None)
            tokenizer_type = getattr(args, 'tokenizer_type', None)

            # Initialize encoder and repo_root before starting curses
            try:
                encoder_and_info = get_encoder(encoding_name, model_name, tokenizer_type)
                repo_root = find_git_root(directory)

                # Now start curses with the pre-initialized encoder
                curses.wrapper(lambda stdscr: tui(stdscr, directory, encoder_and_info, repo_root))
            except ImportError as e:
                # This captures the import errors without trying to use curses
                print(str(e))
                sys.exit(1)

        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            sys.exit(str(e))

if __name__ == "__main__":
    main()
