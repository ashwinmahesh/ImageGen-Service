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
                    for identifier in ('model-name','open-queue','queue-badge','status-text','queue-dialog','pause-queue','queue-list'):
                        self.assertIn(identifier,elements.ids)
                    header=re.search(r'<header\b.*?</header>',response.text,re.S).group()
                    headers.append(re.sub(r'\s+aria-current="page"','',header))
            finally:client.close()
        self.assertEqual(headers[0],headers[1]);self.assertEqual(headers[0],headers[2])

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
