import os
import configparser
from appdirs import user_config_dir

# Configuration directory and file
APP_NAME = "tokdu"
CONFIG_DIR = user_config_dir(APP_NAME)
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")

# Default configuration values
DEFAULT_CONFIG = {
    "tokenizer": {
        "type": "tiktoken",
        "model": "gpt-4o",
        "encoding": "",
    }
}

def ensure_config_dir():
    """Ensure the configuration directory exists."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

def get_config():
    """Load configuration from the config file."""
    ensure_config_dir()

    config = configparser.ConfigParser()
    # Set default values
    for section, options in DEFAULT_CONFIG.items():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in options.items():
            config.set(section, key, str(value))

    # Load existing configuration if it exists
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)

    return config

def save_config(config):
    """Save the configuration to the config file."""
    ensure_config_dir()

    with open(CONFIG_FILE, 'w') as f:
        config.write(f)

def set_config_value(section, key, value):
    """Set a configuration value and save the configuration."""
    config = get_config()

    if not config.has_section(section):
        config.add_section(section)

    # Handle mutually exclusive model and encoding settings
    if section == 'tokenizer' and key in ('model', 'encoding'):
        # Clear the other setting when one is set
        other_key = 'encoding' if key == 'model' else 'model'
        if value:  # Only clear the other if a value is actually being set
            config.set(section, other_key, '')

    config.set(section, key, str(value))
    save_config(config)

    # Return info about what was cleared, if applicable
    if section == 'tokenizer' and key in ('model', 'encoding') and value:
        other_key = 'encoding' if key == 'model' else 'model'
        return other_key
    return None

def get_config_value(section, key, default=None):
    """Get a configuration value."""
    config = get_config()

    if config.has_section(section) and config.has_option(section, key):
        return config.get(section, key)

    return default

def print_config():
    """Print the current configuration."""
    config = get_config()

    print(f"Configuration file: {CONFIG_FILE}")
    print("\nCurrent configuration:")

    for section in config.sections():
        print(f"\n[{section}]")
        for key, value in config.items(section):
            print(f"{key} = {value}")
