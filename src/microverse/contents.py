import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from fps import Module

from jupyverse_api.app import App
from jupyverse_api.auth import Auth
from jupyverse_api.contents import Contents

from jupyverse_api.contents.models import (
    Checkpoint,
    Content,
    CreateContent,
    RenameContent,
    SaveContent,
)


class ContentsModule(Module):
    async def prepare(self) -> None:
        app = await self.get(App)
        auth = await self.get(Auth)

        contents = _Contents(app, auth)
        self.put(contents, Contents)


class _Contents(Contents):
    async def get_root_content(
        self,
        content: int,
        user,
    ):
        return await self.read_content("", bool(content))

    async def read_content(
        self, path: str | Path, get_content: bool, file_format: str | None = None
    ) -> Content:
        if isinstance(path, str):
            path = Path(path)
        content: str | dict | list[dict] | None = None
        if get_content:
            if path.is_dir():
                content = [
                    (await self.read_content(subpath, get_content=False)).model_dump()
                    for subpath in path.iterdir()
                    if not subpath.name.startswith(".")
                ]
        format: str | None = None
        if path.is_dir():
            size = None
            type = "directory"
            format = "json"
            mimetype = None
        elif path.is_file() or path.is_symlink():
            size = get_file_size(path)
            if path.suffix == ".ipynb":
                type = "notebook"
                format = None
                mimetype = None
                if content is not None:
                    nb: dict
                    if file_format == "json":
                        content = cast(dict, content)
                        nb = content
                    else:
                        content = cast(str, content)
                        nb = json.loads(content)
                    for cell in nb["cells"]:
                        if "metadata" not in cell:
                            cell["metadata"] = {}
                        cell["metadata"].update({"trusted": False})
                        if cell["cell_type"] == "code":
                            cell_source = cell["source"]
                            if not isinstance(cell_source, str):
                                cell["source"] = "".join(cell_source)
                    if file_format != "json":
                        content = json.dumps(nb)
            elif path.suffix == ".json":
                type = "json"
                format = "text"
                mimetype = "application/json"
            else:
                type = "file"
                format = None
                mimetype = "text/plain"
        else:
            raise HTTPException(status_code=404, detail="Item not found")
        if path.is_dir():
            size = None
            type = "directory"
            format = "json"
            mimetype = None
        return Content(
            **{
                "name": path.name,
                "path": path.as_posix(),
                "last_modified": get_file_modification_time(path),
                "created": get_file_creation_time(path),
                "content": content,
                "format": format,
                "mimetype": mimetype,
                "size": size,
                "writable": is_file_writable(path),
                "type": type,
            }
        )


    @property
    def file_id_manager(self): pass

    async def write_content(self, content: SaveContent | dict) -> None: pass

    async def create_checkpoint(
        self,
        path,
        user,
    ) -> Checkpoint: pass

    async def create_content(
        self,
        path: str | None,
        request,
        user,
    ) -> Content: pass

    async def get_checkpoint(
        self,
        path,
        user,
    ) -> list[Checkpoint]: return []

    async def get_content(
        self,
        path: str,
        content: int,
        user,
    ) -> Content: pass

    async def save_content(
        self,
        path,
        request,
        response,
        user,
    ) -> Content: pass

    async def delete_content(
        self,
        path,
        user,
    ): pass

    async def rename_content(
        self,
        path,
        request,
        user,
    ) -> Content: pass


def get_file_modification_time(path: Path):
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )


def get_file_creation_time(path: Path):
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )


def get_file_size(path: Path) -> int | None:
    if path.exists():
        return path.stat().st_size
    raise HTTPException(status_code=404, detail="Item not found")


def is_file_writable(path: Path) -> bool:
    if path.exists():
        if path.is_dir():
            # FIXME
            return True
        else:
            return os.access(path, os.W_OK)
    return False
