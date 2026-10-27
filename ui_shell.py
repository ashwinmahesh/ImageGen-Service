"""Render trusted local UI pages with one shared header and queue shell."""
from pathlib import Path
from fastapi.responses import HTMLResponse

PAGES={'index.html':'GENERATOR','gallery.html':'GALLERY','observability.html':'OBSERVABILITY'}


def render_page(project:Path,page:str)->HTMLResponse:
    current=PAGES[page]
    ui=Path(project)/'ui'
    header=(ui/'partials/header.html').read_text()
    for name in PAGES.values():
        header=header.replace(f' __CURRENT_{name}__',' aria-current="page"' if name==current else '')
    content=(ui/page).read_text()
    for marker,partial in [('IMAGEGEN_HEADER',header),('IMAGEGEN_QUEUE',(ui/'partials/queue.html').read_text())]:
        token=f'<!-- {marker} -->'
        if content.count(token)!=1:
            raise ValueError(f'{page} must contain one {marker} placeholder')
        content=content.replace(token,partial)
    return HTMLResponse(content,headers={'Cache-Control':'no-store'})
