from __future__ import annotations

from dataclasses import dataclass

from .shared import python_command


@dataclass
class ScriptCommand:
    parts: list[str]

    @classmethod
    def for_script(cls, script_path: str) -> ScriptCommand:
        return cls(list(python_command(script_path)))

    def add(self, *values: object) -> ScriptCommand:
        self.parts.extend(str(value) for value in values)
        return self

    def option(self, flag: str, value: object) -> ScriptCommand:
        return self.add(flag, value)

    def optional(self, flag: str, value: object | None) -> ScriptCommand:
        if value is not None:
            self.option(flag, value)
        return self

    def flag(self, flag: str, enabled: bool) -> ScriptCommand:
        if enabled:
            self.add(flag)
        return self

    def to_list(self) -> list[str]:
        return list(self.parts)


def script_command(script_path: str) -> ScriptCommand:
    return ScriptCommand.for_script(script_path)
