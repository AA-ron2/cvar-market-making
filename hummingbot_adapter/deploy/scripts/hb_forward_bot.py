"""Hummingbot 2.17.0 imports this file as scripts.hb_forward_bot; the bot is hummingbot_adapter/hb_forward_bot.py.

This folder is mounted read-only at /home/hummingbot/scripts. Its __init__.py makes it a regular package, so
the thesis repository's own scripts/ package (on PYTHONPATH as /repo) cannot shadow it. Only the two classes
are imported: Hummingbot picks the first StrategyV2Base and config subclass among the module's names.
"""

from hummingbot_adapter.hb_forward_bot import HbForwardBot, HbForwardBotConfig  # noqa: F401
