"""Shared page rendering/asset contract; no model startup or inference."""
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import server

PROJECT=Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self,content):
        super().__init__();self.ids=[];self.current=[];self.feed(content)
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if attrs.get('id'):self.ids.append(attrs['id'])
        if attrs.get('aria-current')=='page':self.current.append(attrs.get('href'))


class SharedShellTest(unittest.TestCase):
    def test_every_page_renders_same_header_with_correct_active_link(self):
        headers=[]
        with patch.object(server,'PROJECT_ROOT',PROJECT):
            # Deliberately skip the app lifespan: it loads real runtime resources.
            client=TestClient(server.app,base_url='http://127.0.0.1')
            try:
                for path in ('/','/gallery','/observability'):
                    response=client.get(path)
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(response.headers['cache-control'],'no-store')
                    self.assertNotIn('IMAGEGEN_HEADER',response.text)
                    self.assertNotIn('__CURRENT_',response.text)
                    elements=Elements(response.text)
                    self.assertEqual(elements.current,[path])
                    self.assertEqual(len(elements.ids),len(set(elements.ids)))
                    for identifier in ('model-name','open-queue','queue-badge','queue-processing','status-text','queue-dialog','queue-processing-dialog','queue-list'):
                        self.assertIn(identifier,elements.ids)
                    header=re.search(r'<header\b.*?</header>',response.text,re.S).group()
                    headers.append(re.sub(r'\s+aria-current="page"','',header))
            finally:client.close()
        self.assertEqual(headers[0],headers[1]);self.assertEqual(headers[0],headers[2])

    def test_header_and_modal_queue_switches_are_labelled_and_drive_the_shared_pause_flag(self):
        header=(PROJECT/'ui/partials/header.html').read_text()
        modal=(PROJECT/'ui/partials/queue.html').read_text()
        for markup,identifier in ((header,'queue-processing'),(modal,'queue-processing-dialog')):
            switch=re.search(r'<input id="%s"[^>]*>'%identifier,markup).group()
            for attribute in ('type="checkbox"','role="switch"','class="setting-switch"'):
                self.assertIn(attribute,switch)
            self.assertIn('<label for="%s">Process queue</label>'%identifier,markup)
            self.assertIn('aria-label="Help with Process queue"',markup)
            self.assertIn('Generate immediately and Run Now requests still run while paused',markup)
        # The modal switch replaces the old Pause queue button; Clear waiting and Close stay after it.
        self.assertNotIn('pause-queue',modal)
        self.assertLess(modal.index('queue-processing-dialog'),modal.index('id="clear-queue"'))
        self.assertLess(modal.index('id="clear-queue"'),modal.index('id="close-queue"'))
        self.assertRegex(header,r'id="queue-switch-alert" role="alert"')
        script=(PROJECT/'ui/assets/shell.js').read_text()
        # Both controls send the same PATCH and follow the polled /health queue.paused value.
        self.assertIn("requestJSON('/v1/queue',{method:'PATCH'",script)
        self.assertIn("const queueSwitches=() => [$('queue-processing'),$('queue-processing-dialog')];",script)
        self.assertIn("for (const toggle of queueSwitches()) toggle.addEventListener('change',event => setQueuePaused(!event.target.checked));",script)
        self.assertNotIn('pause-queue',script)
        # A poll requested before a pause/resume must not undo it.
        self.assertIn('pollingQueue=true; const pauseRevision=queuePauseRevision;',script)
        self.assertIn('if (!queueMutating && pauseRevision===queuePauseRevision) showQueuePaused(!!summary.paused)',script)

    def test_disabled_controls_are_not_shown_as_busy_and_new_photos_clear_selections(self):
        styles=(PROJECT/'ui/index.html').read_text()+(PROJECT/'ui/assets/shell.css').read_text()
        # A disabled control is unavailable, not working: no wait cursor on it.
        for rule in re.findall(r'[^{}]*:disabled[^{}]*\{[^}]*\}',styles):self.assertNotIn('cursor:wait',rule)
        page=(PROJECT/'ui/index.html').read_text()
        clear=page[page.index('function clearPhoto()'):page.index("$('choose-photo').addEventListener")]
        # Editing a new source, including a previous edit's result, starts without the old selection.
        for reset in ('maskStrokes=[]','maskUpload=null','savedMaskPath=null','savedMaskImage=null',"$('edit-area').value='whole'"):self.assertIn(reset,clear)

    def test_shared_assets_are_local_uncached_and_paths_are_not_arbitrary(self):
        client=TestClient(server.app,base_url='http://127.0.0.1')
        try:
            for suffix,kind in [('css','text/css'),('js','text/javascript')]:
                response=client.get('/assets/shell.'+suffix)
                self.assertEqual(response.status_code,200)
                self.assertIn(kind,response.headers['content-type'])
                self.assertEqual(response.headers['cache-control'],'no-store')
            self.assertEqual(client.get('/assets/server.py').status_code,404)
        finally:client.close()
