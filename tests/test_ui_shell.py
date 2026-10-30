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
                    for identifier in ('model-name','open-queue','queue-badge','queue-processing','status-text','queue-dialog','pause-queue','queue-list'):
                        self.assertIn(identifier,elements.ids)
                    header=re.search(r'<header\b.*?</header>',response.text,re.S).group()
                    headers.append(re.sub(r'\s+aria-current="page"','',header))
            finally:client.close()
        self.assertEqual(headers[0],headers[1]);self.assertEqual(headers[0],headers[2])

    def test_header_queue_switch_is_labelled_and_drives_the_shared_pause_flag(self):
        header=(PROJECT/'ui/partials/header.html').read_text()
        switch=re.search(r'<input id="queue-processing"[^>]*>',header).group()
        for attribute in ('type="checkbox"','role="switch"','class="setting-switch"'):
            self.assertIn(attribute,switch)
        self.assertIn('<label for="queue-processing">Process queue</label>',header)
        self.assertIn('aria-label="Help with Process queue"',header)
        self.assertIn('Generate immediately and Run Now requests still run while paused',header)
        self.assertRegex(header,r'id="queue-switch-alert" role="alert"')
        script=(PROJECT/'ui/assets/shell.js').read_text()
        # Both controls send the same PATCH and follow the polled /health queue.paused value.
        self.assertIn("requestJSON('/v1/queue',{method:'PATCH'",script)
        self.assertIn("$('queue-processing').addEventListener('change',event => setQueuePaused(!event.target.checked))",script)
        self.assertIn("$('pause-queue').addEventListener('click',() => setQueuePaused(!queuePaused))",script)
        # A poll requested before a pause/resume must not undo it.
        self.assertIn('pollingQueue=true; const pauseRevision=queuePauseRevision;',script)
        self.assertIn('if (!queueMutating && pauseRevision===queuePauseRevision) showQueuePaused(!!summary.paused)',script)

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
